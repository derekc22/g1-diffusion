"""Model definitions for G1 Diffusion."""

# DDPM models
from .stage1_diffusion import (
    Stage1HandDiffusion,
    Stage1HandDiffusionMLP,
    ObjectGeometryEncoder,
)
from .stage2_diffusion import (
    Stage2TransformerModel,
    Stage2MLPModel,
)

# Flow matching models
from .stage1_flow_matching import (
    Stage1HandFlowMatching,
    Stage1HandFlowMatchingMLP,
)
from .stage2_flow_matching import (
    Stage2FMTransformerModel,
    Stage2FMMLPModel,
)

# HuggingFace dataset models
from .stage1_hf_diffusion import (
    Stage1HFHandDiffusion,
    Stage1HFHandDiffusionMLP,
    ObjectMotionEncoder,
)

__all__ = [
    # DDPM
    "Stage1HandDiffusion",
    "Stage1HandDiffusionMLP",
    "ObjectGeometryEncoder",
    "Stage2TransformerModel",
    "Stage2MLPModel",
    # Flow Matching
    "Stage1HandFlowMatching",
    "Stage1HandFlowMatchingMLP",
    "Stage2FMTransformerModel",
    "Stage2FMMLPModel",
    # HuggingFace dataset
    "Stage1HFHandDiffusion",
    "Stage1HFHandDiffusionMLP",
    "ObjectMotionEncoder",
]
