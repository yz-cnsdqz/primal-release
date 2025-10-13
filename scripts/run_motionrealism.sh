#!/bin/bash

###############################################################################
# Motion Realism Evaluation Pipeline
# 
# This script provides a complete pipeline for:
# 1. Generating motion sequences from trained models
# 2. Rendering generated motions as videos
# 3. Evaluating motion quality metrics
#
# Usage:
#   bash scripts/run_motionrealism.sh <mode> <checkpoint_dir>
#
# Modes:
#   gen    - Generate motion sequences
#   render - Render videos from generated sequences  
#   eval   - Evaluate motion quality metrics
#
# Example:
#   bash scripts/run_motionrealism.sh gen logs/primal/motion_diffuser_ar_30fps/runs/2025-02-01_05-14-50_silu_anchor
###############################################################################

set -e  # Exit on error

###############################################################################
# Python Environment Detection
###############################################################################
detect_python_command() {
    # Check if we're in a Poetry environment
    if command -v poetry >/dev/null 2>&1 && [ -f "pyproject.toml" ]; then
        echo "poetry run python"
        return
    fi
    
    # Check if we're in a conda environment
    if [ -n "$CONDA_DEFAULT_ENV" ] && command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi
    
    # Check for standard python3/python
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return
    elif command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi
    
    echo "Error: No Python interpreter found!" >&2
    echo "Please ensure one of the following is available:" >&2
    echo "  - Poetry (with pyproject.toml)" >&2
    echo "  - Conda environment with Python" >&2
    echo "  - Standard python/python3 installation" >&2
    exit 1
}

# Configuration
readonly DATASETS=(
    "HumanEva"
    "SFU"
)

readonly GENERATION_PARAMS="--ckptidx 29999 --use_ema --use_inertialization --use_reproj_kpts --frame_skip 150"
readonly OUTPUT_DIR="outputs/MotionRealism"

# Detect Python command
readonly PYTHON_CMD=$(detect_python_command)

# Validate arguments
if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: $0 <mode> <checkpoint_directory>"
    echo ""
    echo "Available modes:"
    echo "  gen    - Generate motion sequences"
    echo "  render - Render videos from generated sequences"
    echo "  eval   - Evaluate motion quality metrics"
    echo ""
    echo "Example:"
    echo "  $0 gen logs/primal/motion_diffuser_ar_30fps/runs/2025-02-01_05-14-50_silu_anchor"
    exit 1
fi

readonly MODE="$1"
readonly CHECKPOINT_DIR="$2"
readonly EXPERIMENT_NAME=$(basename "$CHECKPOINT_DIR")

# Validate checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "========================================="
echo "Motion Realism Pipeline"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Python Command: $PYTHON_CMD"
echo "========================================="

###############################################################################
# Generation Mode
###############################################################################
generate_motions() {
    echo "Starting motion generation..."
    
    for dataset in "${DATASETS[@]}"; do
        echo "Generating motions for dataset: $dataset"
        
        $PYTHON_CMD experiments/gen_Motion.py \
            --expdir "$CHECKPOINT_DIR" \
            --dataset "$dataset" \
            $GENERATION_PARAMS
        
        echo "✓ Completed generation for $dataset"
    done
    
    echo "All motion generation completed!"
}

###############################################################################
# Rendering Mode  
###############################################################################
render_videos() {
    echo "Starting video rendering..."
    
    for dataset in "${DATASETS[@]}"; do
        echo "Rendering videos for dataset: $dataset"
        
        local pkl_file="$OUTPUT_DIR/$EXPERIMENT_NAME/${dataset}_ema-True_reproj-True_inertial-True.pkl"
        
        if [ ! -f "$pkl_file" ]; then
            echo "Warning: PKL file not found: $pkl_file"
            echo "Skipping rendering for $dataset"
            continue
        fi
        
        $PYTHON_CMD primal/rendering/render.py "$pkl_file" false
        echo "✓ Completed rendering for $dataset"
    done
    
    echo "All video rendering completed!"
}

###############################################################################
# Evaluation Mode
###############################################################################
evaluate_metrics() {
    echo "Starting motion evaluation..."
    
    for dataset in "${DATASETS[@]}"; do
        echo "Evaluating metrics for dataset: $dataset"
        
        local pkl_file="$OUTPUT_DIR/$EXPERIMENT_NAME/${dataset}_ema-True_reproj-True_inertial-True.pkl"
        local output_file="$OUTPUT_DIR/$EXPERIMENT_NAME/${dataset}_ema-True_reproj-True_inertial-True.txt"
        
        if [ ! -f "$pkl_file" ]; then
            echo "Warning: PKL file not found: $pkl_file"
            echo "Skipping evaluation for $dataset"
            continue
        fi
        
        echo "Evaluating: $pkl_file"
        echo "Output: $output_file"
        
        $PYTHON_CMD experiments/eval_generation.py "$pkl_file" > "$output_file"
        echo "✓ Completed evaluation for $dataset"
    done
    
    echo "All motion evaluation completed!"
    echo ""
    echo "Evaluation results saved in:"
    for dataset in "${DATASETS[@]}"; do
        local output_file="$OUTPUT_DIR/$EXPERIMENT_NAME/${dataset}_ema-True_reproj-True_inertial-True.txt"
        if [ -f "$output_file" ]; then
            echo "  - $output_file"
        fi
    done
}

###############################################################################
# Main Execution
###############################################################################
case "$MODE" in
    "gen")
        generate_motions
        ;;
    "render")
        render_videos
        ;;
    "eval")
        evaluate_metrics
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Available modes: gen, render, eval"
        exit 1
        ;;
esac

echo ""
echo "✅ Operation '$MODE' completed successfully!"
echo "Experiment: $EXPERIMENT_NAME"
echo "========================================="