#!/bin/bash
# Export and optimize diffusion models for deployment
#
# Usage:
#   ./export_optimized.sh <checkpoint_path> <output_dir> [options]
#
# Options:
#   --onnx          Export to ONNX format
#   --tensorrt      Convert to TensorRT (requires TensorRT)
#   --fp16          Use FP16 precision
#   --benchmark     Run inference benchmarks
#
# Example:
#   ./export_optimized.sh logs/exp1/checkpoints/best.pt exports/ --onnx --fp16

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CKPT_PATH=""
OUTPUT_DIR="./exports"
DO_ONNX=false
DO_TENSORRT=false
USE_FP16=false
DO_BENCHMARK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --onnx)
            DO_ONNX=true
            shift
            ;;
        --tensorrt)
            DO_TENSORRT=true
            shift
            ;;
        --fp16)
            USE_FP16=true
            shift
            ;;
        --benchmark)
            DO_BENCHMARK=true
            shift
            ;;
        -h|--help)
            echo "Export and optimize diffusion models for deployment"
            echo ""
            echo "Usage: $0 <checkpoint_path> <output_dir> [options]"
            echo ""
            echo "Options:"
            echo "  --onnx          Export to ONNX format"
            echo "  --tensorrt      Convert to TensorRT"
            echo "  --fp16          Use FP16 precision"
            echo "  --benchmark     Run inference benchmarks"
            exit 0
            ;;
        *)
            if [ -z "$CKPT_PATH" ]; then
                CKPT_PATH="$1"
            elif [ -z "$OUTPUT_DIR" ] || [ "$OUTPUT_DIR" == "./exports" ]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$CKPT_PATH" ]; then
    echo "Error: checkpoint path required"
    echo "Usage: $0 <checkpoint_path> <output_dir> [options]"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Model Export & Optimization"
echo "=========================================="
echo "Checkpoint: $CKPT_PATH"
echo "Output: $OUTPUT_DIR"
echo "ONNX: $DO_ONNX"
echo "TensorRT: $DO_TENSORRT"
echo "FP16: $USE_FP16"
echo "Benchmark: $DO_BENCHMARK"
echo "=========================================="

cd "$PROJECT_ROOT"

# Run benchmark if requested
if [ "$DO_BENCHMARK" = true ]; then
    echo ""
    echo "Running benchmarks..."
    python scripts/sample_stage1_optimized.py \
        --benchmark \
        --config ./config/sample_stage1.yaml
fi

# Export to ONNX if requested
if [ "$DO_ONNX" = true ]; then
    echo ""
    echo "Exporting to ONNX..."
    python scripts/sample_stage1_optimized.py \
        --export-onnx "$OUTPUT_DIR/onnx" \
        --config ./config/sample_stage1.yaml
    
    # Try to optimize ONNX if onnxruntime is available
    if python -c "import onnxruntime" 2>/dev/null; then
        echo "Optimizing ONNX model..."
        python -c "
from utils.inference_optimization import ONNXExporter
import os

onnx_path = os.path.join('$OUTPUT_DIR', 'onnx', 'stage1_full.onnx')
if os.path.exists(onnx_path):
    ONNXExporter.optimize_for_inference(onnx_path)
"
    fi
fi

# Convert to TensorRT if requested
if [ "$DO_TENSORRT" = true ]; then
    echo ""
    echo "Converting to TensorRT..."
    
    PRECISION="fp32"
    if [ "$USE_FP16" = true ]; then
        PRECISION="fp16"
    fi
    
    python -c "
from utils.inference_optimization import TensorRTConverter
import os

onnx_path = os.path.join('$OUTPUT_DIR', 'onnx', 'stage1_full.onnx')
trt_path = os.path.join('$OUTPUT_DIR', 'tensorrt', 'stage1_$PRECISION.trt')

os.makedirs(os.path.dirname(trt_path), exist_ok=True)

if os.path.exists(onnx_path):
    TensorRTConverter.convert_to_tensorrt(
        onnx_path,
        trt_path,
        precision='$PRECISION',
    )
else:
    print('ONNX model not found. Run with --onnx first.')
"
fi

echo ""
echo "=========================================="
echo "Export complete!"
echo "=========================================="
echo "Output files:"
ls -la "$OUTPUT_DIR"
