"""
Inference Time Optimization Utilities for G1 Diffusion Models

This module provides various optimization techniques to accelerate inference:
1. Mixed precision (FP16/BF16) inference with automatic mixed precision (AMP)
2. torch.compile() for graph optimization (PyTorch 2.0+)
3. DDIM fast sampling (reduces 1000 steps to 20-50)
4. Dynamic quantization (INT8) for CPU deployment
5. ONNX export for TensorRT/ONNX Runtime deployment
6. Model caching and warmup utilities

Reference: theia-tiny-patch16-224-cddsv (IsaacLab integration)
- Uses bfloat16 for negligible overhead during policy training
- Lightweight model design with efficient inference patterns
"""

from __future__ import annotations
import os
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable, Tuple, List, Union
from contextlib import contextmanager
import functools

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np


class PrecisionMode(Enum):
    """Supported precision modes for inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"  # Dynamic quantization


class SamplerType(Enum):
    """Supported diffusion samplers."""
    DDPM = "ddpm"  # Original, 1000 steps
    DDIM = "ddim"  # Fast, 20-100 steps
    DPM_SOLVER = "dpm_solver"  # Very fast, 10-25 steps


@dataclass
class InferenceConfig:
    """Configuration for optimized inference."""
    # Precision settings
    precision: PrecisionMode = PrecisionMode.FP16
    
    # Compilation settings (PyTorch 2.0+)
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = False  # Set True for better optimization, may fail on dynamic models
    
    # Sampling settings
    sampler: SamplerType = SamplerType.DDIM
    num_inference_steps: int = 50  # For DDIM/DPM-Solver
    ddim_eta: float = 0.0  # 0.0 = deterministic, 1.0 = full stochasticity
    
    # Memory optimization
    enable_gradient_checkpointing: bool = False
    channels_last: bool = False  # Memory format optimization for CNN-heavy models
    
    # Warmup
    warmup_iterations: int = 3
    
    # Export settings
    onnx_opset_version: int = 17


class OptimizedInferenceWrapper:
    """
    Wrapper for optimized model inference following theia patterns.
    
    Applies various optimization techniques while maintaining model quality.
    Key optimizations from theia-tiny-patch16-224-cddsv:
    - bfloat16 inference with negligible overhead
    - torch.inference_mode() context
    - Efficient batching
    
    Usage:
        wrapper = OptimizedInferenceWrapper(model, config)
        wrapper.warmup(sample_input)
        output = wrapper(x, t, cond)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Apply precision mode
        self._apply_precision_mode()
        
        # Apply torch.compile if available and requested
        self._apply_compilation()
        
        # Memory format optimization
        if self.config.channels_last and torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.model.eval()
        self._is_warmed_up = False
    
    def _apply_precision_mode(self):
        """Apply the configured precision mode to the model."""
        if self.config.precision == PrecisionMode.FP16:
            if self.device.type == "cuda":
                self.model = self.model.half()
                self.dtype = torch.float16
            else:
                warnings.warn("FP16 requested but CUDA not available, using FP32")
                self.dtype = torch.float32
                
        elif self.config.precision == PrecisionMode.BF16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.model = self.model.to(torch.bfloat16)
                self.dtype = torch.bfloat16
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                # Apple Silicon support
                self.model = self.model.to(torch.bfloat16)
                self.dtype = torch.bfloat16
            else:
                warnings.warn("BF16 not supported, using FP32")
                self.dtype = torch.float32
                
        elif self.config.precision == PrecisionMode.INT8:
            # Dynamic quantization for CPU
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.TransformerEncoderLayer},
                dtype=torch.qint8
            )
            self.dtype = torch.float32  # Input still FP32, weights are INT8
            
        else:
            self.dtype = torch.float32
    
    def _apply_compilation(self):
        """Apply torch.compile if available and requested."""
        if not self.config.use_torch_compile:
            return
            
        if not hasattr(torch, 'compile'):
            warnings.warn("torch.compile not available (requires PyTorch 2.0+)")
            return
        
        try:
            self.model = torch.compile(
                self.model,
                mode=self.config.compile_mode,
                fullgraph=self.config.compile_fullgraph,
            )
            print(f"Model compiled with mode='{self.config.compile_mode}'")
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}. Continuing without compilation.")
    
    def warmup(
        self,
        sample_inputs: Tuple[torch.Tensor, ...],
        iterations: Optional[int] = None,
    ):
        """
        Warmup the model to trigger JIT compilation and optimize CUDA kernels.
        
        Args:
            sample_inputs: Tuple of sample input tensors matching model signature
            iterations: Number of warmup iterations (default: config.warmup_iterations)
        """
        iterations = iterations or self.config.warmup_iterations
        
        with torch.inference_mode():
            for _ in range(iterations):
                # Convert inputs to appropriate dtype and device
                inputs = tuple(
                    x.to(self.device, dtype=self.dtype if x.dtype.is_floating_point else x.dtype)
                    for x in sample_inputs
                )
                _ = self.model(*inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self._is_warmed_up = True
        print(f"Model warmed up with {iterations} iterations")
    
    @torch.inference_mode()
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with optimized inference.
        
        Automatically handles:
        - Type conversion to configured precision
        - Device placement
        - Mixed precision context (for FP16/BF16)
        """
        # Convert inputs
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                target_dtype = self.dtype if arg.dtype.is_floating_point else arg.dtype
                processed_args.append(arg.to(self.device, dtype=target_dtype))
            else:
                processed_args.append(arg)
        
        processed_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                target_dtype = self.dtype if v.dtype.is_floating_point else v.dtype
                processed_kwargs[k] = v.to(self.device, dtype=target_dtype)
            else:
                processed_kwargs[k] = v
        
        # Run inference
        if self.config.precision in (PrecisionMode.FP16, PrecisionMode.BF16):
            with autocast(dtype=self.dtype, enabled=self.device.type == "cuda"):
                output = self.model(*processed_args, **processed_kwargs)
        else:
            output = self.model(*processed_args, **processed_kwargs)
        
        return output


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) fast sampler.
    
    Reduces inference from 1000 steps to 20-50 steps with minimal quality loss.
    
    Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        eta: float = 0.0,  # 0 = deterministic
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # Compute beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Compute timestep schedule for DDIM
        self.timesteps = self._compute_timesteps()
    
    def _compute_timesteps(self) -> torch.Tensor:
        """Compute evenly spaced timesteps for DDIM sampling."""
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, self.num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0).long()  # Reverse for denoising
        return timesteps
    
    def to(self, device: torch.device) -> "DDIMSampler":
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self
    
    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition_fn: Callable[..., Tuple[torch.Tensor, ...]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Run DDIM sampling.
        
        Args:
            model: Denoising model that predicts x0
            shape: Shape of output tensor (B, T, D)
            condition_fn: Function that returns conditioning tensors
            device: Target device
            dtype: Data type for computation
            
        Returns:
            Sampled tensor of shape `shape`
        """
        batch_size = shape[0]
        
        # Start from random noise
        x = torch.randn(shape, device=device, dtype=dtype)
        
        # Get conditioning
        cond = condition_fn()
        
        # DDIM sampling loop
        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict x0
            if isinstance(cond, tuple):
                x0_pred = model(x, t_batch, *cond)
            else:
                x0_pred = model(x, t_batch, cond)
            
            if i < len(self.timesteps) - 1:
                # Compute variance schedule parameters
                alpha_cumprod_t = self.alphas_cumprod[t]
                alpha_cumprod_t_prev = self.alphas_cumprod[self.timesteps[i + 1]]
                
                # DDIM update
                # Predicted noise
                pred_noise = (x - torch.sqrt(alpha_cumprod_t) * x0_pred) / torch.sqrt(1 - alpha_cumprod_t)
                
                # Variance
                sigma = self.eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * pred_noise
                
                # Compute x_{t-1}
                x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + dir_xt
                
                if self.eta > 0:
                    noise = torch.randn_like(x)
                    x = x + sigma * noise
            else:
                x = x0_pred
        
        return x


class DPMSolverSampler:
    """
    DPM-Solver fast sampler for even faster inference (10-25 steps).
    
    Reference: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling" 
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        solver_order: int = 2,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.solver_order = solver_order
        
        # Compute schedules
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Log SNR for DPM-Solver
        self.log_snr = torch.log(self.alphas_cumprod / (1 - self.alphas_cumprod))
        
        self.timesteps = self._compute_timesteps()
    
    def _compute_timesteps(self) -> torch.Tensor:
        """Compute timesteps using logSNR spacing."""
        log_snr_max = self.log_snr[0]
        log_snr_min = self.log_snr[-1]
        log_snr_steps = torch.linspace(log_snr_max, log_snr_min, self.num_inference_steps + 1)
        
        # Find nearest timesteps
        timesteps = []
        for log_snr_val in log_snr_steps[:-1]:
            idx = (self.log_snr - log_snr_val).abs().argmin()
            timesteps.append(idx.item())
        
        return torch.tensor(timesteps, dtype=torch.long)
    
    def to(self, device: torch.device) -> "DPMSolverSampler":
        """Move scheduler tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.log_snr = self.log_snr.to(device)
        self.timesteps = self.timesteps.to(device)
        return self
    
    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition_fn: Callable[..., Tuple[torch.Tensor, ...]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Run DPM-Solver sampling with second-order updates."""
        batch_size = shape[0]
        x = torch.randn(shape, device=device, dtype=dtype)
        cond = condition_fn()
        
        # Store model outputs for multistep
        model_outputs = []
        
        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict x0
            if isinstance(cond, tuple):
                x0_pred = model(x, t_batch, *cond)
            else:
                x0_pred = model(x, t_batch, cond)
            
            # Convert to noise prediction for DPM-Solver
            alpha_t = self.alphas_cumprod[t]
            noise_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)
            model_outputs.append(noise_pred)
            
            if i < len(self.timesteps) - 1:
                t_next = self.timesteps[i + 1]
                alpha_t_next = self.alphas_cumprod[t_next]
                
                # DPM-Solver-2 update (simplified)
                if self.solver_order == 2 and len(model_outputs) >= 2:
                    # Second order update using previous output
                    noise_pred_prev = model_outputs[-2]
                    noise_pred = 1.5 * noise_pred - 0.5 * noise_pred_prev
                
                # Compute x_next
                x = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
                
                # Keep only last few outputs for memory
                if len(model_outputs) > self.solver_order:
                    model_outputs = model_outputs[-self.solver_order:]
            else:
                x = x0_pred
        
        return x


def create_sampler(
    sampler_type: SamplerType,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    **kwargs,
) -> Union[DDIMSampler, DPMSolverSampler]:
    """Factory function to create the appropriate sampler."""
    if sampler_type == SamplerType.DDIM:
        return DDIMSampler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            eta=kwargs.get("ddim_eta", 0.0),
        )
    elif sampler_type == SamplerType.DPM_SOLVER:
        return DPMSolverSampler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            solver_order=kwargs.get("solver_order", 2),
        )
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


class ONNXExporter:
    """
    ONNX exporter for diffusion models.
    
    Follows theia practices for deployment optimization:
    - Separate export of encoder and denoiser
    - Dynamic axes for variable sequence lengths
    - Optimization for TensorRT/ONNX Runtime
    """
    
    @staticmethod
    def export_stage1_model(
        model: nn.Module,
        output_dir: str,
        sample_inputs: Dict[str, torch.Tensor],
        opset_version: int = 17,
    ):
        """
        Export Stage 1 model to ONNX format.
        
        Args:
            model: Stage1HandDiffusion model
            output_dir: Directory to save ONNX files
            sample_inputs: Dict with sample tensors for tracing
            opset_version: ONNX opset version
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model.eval()
        device = next(model.parameters()).device
        
        # Sample inputs
        x = sample_inputs["x"].to(device)
        t = sample_inputs["t"].to(device)
        bps = sample_inputs["bps"].to(device)
        centroid = sample_inputs["centroid"].to(device)
        
        B, T, D = x.shape
        
        # Export full model
        full_path = os.path.join(output_dir, "stage1_full.onnx")
        
        torch.onnx.export(
            model,
            (x, t, bps, centroid),
            full_path,
            input_names=["noisy_hands", "timestep", "bps_encoding", "object_centroid"],
            output_names=["predicted_hands"],
            dynamic_axes={
                "noisy_hands": {0: "batch", 1: "seq_len"},
                "timestep": {0: "batch"},
                "bps_encoding": {0: "batch", 1: "seq_len"},
                "object_centroid": {0: "batch", 1: "seq_len"},
                "predicted_hands": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        print(f"Exported full model to {full_path}")
        
        # Optionally export encoder separately for caching
        encoder_path = os.path.join(output_dir, "stage1_encoder.onnx")
        
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            
            def forward(self, bps, centroid):
                return self.encoder(bps, centroid)
        
        encoder_wrapper = EncoderWrapper(model.encoder).to(device)
        
        torch.onnx.export(
            encoder_wrapper,
            (bps, centroid),
            encoder_path,
            input_names=["bps_encoding", "object_centroid"],
            output_names=["object_features"],
            dynamic_axes={
                "bps_encoding": {0: "batch", 1: "seq_len"},
                "object_centroid": {0: "batch", 1: "seq_len"},
                "object_features": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        print(f"Exported encoder to {encoder_path}")
        
        return full_path, encoder_path
    
    @staticmethod
    def optimize_for_inference(
        onnx_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Optimize ONNX model for inference using ONNX Runtime.
        
        Args:
            onnx_path: Path to input ONNX file
            output_path: Path for optimized model (default: adds '_optimized' suffix)
            
        Returns:
            Path to optimized model
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
        except ImportError:
            warnings.warn("onnx or onnxruntime not installed. Skipping optimization.")
            return onnx_path
        
        if output_path is None:
            base, ext = os.path.splitext(onnx_path)
            output_path = f"{base}_optimized{ext}"
        
        # Load and optimize
        model = onnx.load(onnx_path)
        
        # Basic optimization
        from onnx import optimizer as onnx_optimizer
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
        ]
        
        try:
            optimized_model = onnx_optimizer.optimize(model, passes)
            onnx.save(optimized_model, output_path)
            print(f"Optimized model saved to {output_path}")
        except Exception as e:
            warnings.warn(f"ONNX optimization failed: {e}")
            output_path = onnx_path
        
        return output_path


class TensorRTConverter:
    """
    TensorRT conversion utilities for maximum inference speed.
    
    Provides FP16 and INT8 quantization for NVIDIA GPUs.
    """
    
    @staticmethod
    def convert_to_tensorrt(
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",  # "fp32", "fp16", "int8"
        max_batch_size: int = 1,
        max_workspace_size: int = 1 << 30,  # 1GB
    ) -> Optional[str]:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path for TensorRT engine
            precision: Target precision ("fp32", "fp16", "int8")
            max_batch_size: Maximum batch size for engine
            max_workspace_size: Maximum workspace memory
            
        Returns:
            Path to TensorRT engine or None if conversion fails
        """
        try:
            import tensorrt as trt
        except ImportError:
            warnings.warn("TensorRT not installed. Skipping conversion.")
            return None
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # Note: INT8 requires calibration data
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                warnings.warn("Failed to build TensorRT engine")
                return None
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"TensorRT engine saved to {output_path}")
            return output_path


def benchmark_inference(
    model: nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_cuda_events: bool = True,
) -> Dict[str, float]:
    """
    Benchmark model inference time.
    
    Args:
        model: Model to benchmark
        sample_inputs: Sample input tensors
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        use_cuda_events: Use CUDA events for precise timing
        
    Returns:
        Dict with timing statistics (mean, std, min, max in ms)
    """
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    
    model.eval()
    
    # Move inputs to device
    inputs = tuple(x.to(device) for x in sample_inputs)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = model(*inputs)
    
    if is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    
    with torch.inference_mode():
        for _ in range(num_iterations):
            if is_cuda and use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = model(*inputs)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                start = time.perf_counter()
                _ = model(*inputs)
                if is_cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "throughput_hz": float(1000.0 / np.mean(times)),
    }


def get_optimal_config(
    device: torch.device,
    model_size: str = "base",  # "small", "base", "large"
    target_latency_ms: Optional[float] = None,
) -> InferenceConfig:
    """
    Get optimal inference configuration based on hardware and requirements.
    
    Args:
        device: Target device
        model_size: Model size category
        target_latency_ms: Target latency constraint (optional)
        
    Returns:
        Optimized InferenceConfig
    """
    config = InferenceConfig()
    
    # Device-specific optimizations
    if device.type == "cuda":
        # Check for Ampere+ (SM >= 8.0) for BF16 support
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                config.precision = PrecisionMode.BF16
            else:
                config.precision = PrecisionMode.FP16
        
        config.use_torch_compile = True
        config.compile_mode = "reduce-overhead"
        
    elif device.type == "mps":
        # Apple Silicon
        config.precision = PrecisionMode.FP16
        config.use_torch_compile = False  # Not well supported on MPS yet
        
    else:
        # CPU
        config.precision = PrecisionMode.INT8
        config.use_torch_compile = True
        config.compile_mode = "default"
    
    # Adjust sampling steps based on target latency
    if target_latency_ms is not None:
        # Rough heuristic: larger models need more aggressive step reduction
        if model_size == "small":
            base_steps = 50
        elif model_size == "base":
            base_steps = 30
        else:
            base_steps = 20
        
        config.num_inference_steps = max(10, min(base_steps, int(target_latency_ms / 10)))
        config.sampler = SamplerType.DPM_SOLVER if config.num_inference_steps <= 25 else SamplerType.DDIM
    
    return config
