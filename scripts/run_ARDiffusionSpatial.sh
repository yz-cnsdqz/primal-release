#!/bin/bash

###############################################################################
# ARDiffusion Spatial Control Pipeline
#
# This script provides a complete pipeline for:
# 1. Generating spatial target-based motion sequences from trained models
# 2. Rendering generated motions as videos
# 3. Evaluating motion quality metrics
#
# Usage:
#   bash scripts/run_ARDiffusionSpatial.sh <mode> [checkpoint_dir]
#
# Modes:
#   gen    - Generate motion sequences for all spatial targets
#   render - Render videos from generated sequences
#   eval   - Evaluate motion quality metrics
#
# Example:
#   bash scripts/run_ARDiffusionSpatial.sh gen logs/primal/motion_diffuser_ar_spatial/runs/2025-02-23_15-44-35
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

readonly TARGET_DIRS=(0 90 180 270)
readonly TARGET_DISTS=(1.5 1.0)

readonly GENERATION_PARAMS="--ckptidx 99 --use_ema --dataset SFU --use_inertialization --use_reproj_kpts --frame_skip 400"
readonly OUTPUT_DIR="outputs/Res_ARDiffusionSpatial"

# Detect Python command
readonly PYTHON_CMD=$(detect_python_command)

# Validate arguments
if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: $0 <mode> <checkpoint_directory>"
    echo ""
    echo "Available modes:"
    echo "  gen    - Generate motion sequences for all spatial targets"
    echo "  render - Render videos from generated sequences"
    echo "  eval   - Evaluate motion quality metrics"
    echo ""
    echo "Example:"
    echo "  $0 gen logs/primal/motion_diffuser_ar_spatial/runs/2025-02-23_15-44-35"
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

readonly EXPERIMENTS=("$CHECKPOINT_DIR")

echo "========================================="
echo "ARDiffusion Spatial Control Pipeline"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Target Directions: ${TARGET_DIRS[*]}"
echo "Target dists: ${TARGET_DISTS[*]}"
echo "Python Command: $PYTHON_CMD"
echo "========================================="

###############################################################################
# Generation Mode
###############################################################################
generate_motions() {
    echo "Starting spatial target motion generation..."

    local total_combinations=$((${#EXPERIMENTS[@]} * ${#TARGET_DIRS[@]} * ${#TARGET_DISTS[@]}))
    local current=0

    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Processing experiment: $exp_name"

        for dir in "${TARGET_DIRS[@]}"; do
            for dist in "${TARGET_DISTS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Generating dir=$dir, dist=$dist for $exp_name"

                $PYTHON_CMD experiments/gen_Motion_ARDiffusionSpatial.py \
                    --expdir "$expdir" \
                    --target_dir "$dir" \
                    --target_dist "$dist" \
                    $GENERATION_PARAMS

                echo "✓ Completed generation for dir=$dir, dist=$dist"
            done
        done

        echo "✓ Completed all spatial targets for $exp_name"
    done

    echo "All spatial target motion generation completed!"
}

###############################################################################
# Rendering Mode
###############################################################################
render_videos() {
    echo "Starting video rendering..."

    local total_combinations=$((${#EXPERIMENTS[@]} * ${#TARGET_DIRS[@]} * ${#TARGET_DISTS[@]}))
    local current=0

    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Rendering videos for experiment: $exp_name"

        for dir in "${TARGET_DIRS[@]}"; do
            for dist in "${TARGET_DISTS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Rendering dir=$dir, dist=$dist for $exp_name"

                local formatted_dir="$dir.0"
                local formatted_dist="$dist"
                local pkl_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_dist-${formatted_dist}.pkl"

                if [ ! -f "$pkl_file" ]; then
                    echo "Warning: PKL file not found: $pkl_file"
                    echo "Skipping rendering for dir=$dir, dist=$dist"
                    continue
                fi

                $PYTHON_CMD primal/rendering/render.py "$pkl_file" false
                echo "✓ Completed rendering for dir=$dir, dist=$dist"
            done
        done

        echo "✓ Completed all renderings for $exp_name"
    done

    echo "All video rendering completed!"
}

###############################################################################
# Evaluation Mode
###############################################################################
evaluate_metrics() {
    echo "Starting motion evaluation..."

    local total_combinations=$((${#EXPERIMENTS[@]} * ${#TARGET_DIRS[@]} * ${#TARGET_DISTS[@]}))
    local current=0

    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Evaluating metrics for experiment: $exp_name"

        for dir in "${TARGET_DIRS[@]}"; do
            for dist in "${TARGET_DISTS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Evaluating dir=$dir, dist=$dist for $exp_name"

                local formatted_dir="$dir.0"
                local formatted_dist="$dist"
                local pkl_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_dist-${formatted_dist}.pkl"
                local output_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_dist-${formatted_dist}.txt"

                if [ ! -f "$pkl_file" ]; then
                    echo "Warning: PKL file not found: $pkl_file"
                    echo "Skipping evaluation for dir=$dir, dist=$dist"
                    continue
                fi

                echo "Evaluating: $pkl_file"
                echo "Output: $output_file"

                $PYTHON_CMD experiments/eval_generation.py "$pkl_file" > "$output_file"
                echo "✓ Completed evaluation for dir=$dir, dist=$dist"
            done
        done

        echo "✓ Completed all evaluations for $exp_name"
    done

    echo "All motion evaluation completed!"
    echo ""
    echo "Evaluation results saved in:"
    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        for dir in "${TARGET_DIRS[@]}"; do
            for dist in "${TARGET_DISTS[@]}"; do
                local formatted_dir="$dir.0"
                local formatted_dist="$dist"
                local output_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_dist-${formatted_dist}.txt"
                if [ -f "$output_file" ]; then
                    echo "  - $output_file"
                fi
            done
        done
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
echo "Spatial targets processed: ${#TARGET_DIRS[@]} directions × ${#TARGET_DISTS[@]} dists"
echo "========================================="
