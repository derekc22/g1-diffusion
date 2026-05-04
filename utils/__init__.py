"""Utility functions for G1 Diffusion."""

from .inference_optimization import (
    InferenceConfig,
    PrecisionMode,
    SamplerType,
    OptimizedInferenceWrapper,
    DDIMSampler,
    DPMSolverSampler,
    create_sampler,
    benchmark_inference,
    get_optimal_config,
    ONNXExporter,
    TensorRTConverter,
)

from .flow_matching import (
    FlowMatchingConfig,
    FlowMatchingSchedule,
    euler_solve,
    midpoint_solve,
    rk4_solve,
    get_ode_solver,
    flow_matching_timestep_embedding,
)
from .object_conditioning import (
    OBJECT_CONDITIONING_VARIANTS,
    apply_object_conditioning_variant,
    apply_temporal_conditioning_variant,
    describe_object_conditioning_variant,
    normalize_object_conditioning_variant,
)

__all__ = [
    # Inference optimization
    "InferenceConfig",
    "PrecisionMode", 
    "SamplerType",
    "OptimizedInferenceWrapper",
    "DDIMSampler",
    "DPMSolverSampler",
    "create_sampler",
    "benchmark_inference",
    "get_optimal_config",
    "ONNXExporter",
    "TensorRTConverter",
    # Flow matching
    "FlowMatchingConfig",
    "FlowMatchingSchedule",
    "euler_solve",
    "midpoint_solve",
    "rk4_solve",
    "get_ode_solver",
    "flow_matching_timestep_embedding",
    # Object conditioning variants
    "OBJECT_CONDITIONING_VARIANTS",
    "apply_object_conditioning_variant",
    "apply_temporal_conditioning_variant",
    "describe_object_conditioning_variant",
    "normalize_object_conditioning_variant",
]
