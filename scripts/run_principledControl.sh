#!/bin/bash

###############################################################################
# Principled Spatial Control Pipeline
# 
# This script provides a complete pipeline for:
# 1. Generating spatial-controlled motion sequences from trained models
# 2. Rendering generated motions as videos
# 3. Evaluating motion quality metrics
#
# Usage:
#   bash scripts/run_principledControl.sh <mode> [checkpoint_dir]
#
# Modes:
#   gen    - Generate motion sequences for all direction/speed combinations
#   render - Render videos from generated sequences  
#   eval   - Evaluate motion quality metrics
#
# Example:
#   bash scripts/run_principledControl.sh gen logs/primal/motion_diffuser_ar_30fps/runs/2025-02-01_05-14-50_silu_anchor
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

readonly DIRECTIONS=(0 90 180 270)  # degrees
readonly SPEEDS=(1.0 4.0)           # m/s

readonly GENERATION_PARAMS="--ckptidx 29999 --use_ema --dataset SFU --use_inertialization --use_reproj_kpts --frame_skip 400"
readonly OUTPUT_DIR="outputs/PrincipledSpatialControl"

# Detect Python command
readonly PYTHON_CMD=$(detect_python_command)

# Validate arguments
if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments"
    echo "Usage: $0 <mode> <checkpoint_directory>"
    echo ""
    echo "Available modes:"
    echo "  gen    - Generate motion sequences for all direction/speed combinations"
    echo "  render - Render videos from generated sequences"
    echo "  eval   - Evaluate motion quality metrics"
    echo ""
    echo "Control Parameters:"
    echo "  Directions: ${DIRECTIONS[*]}° (0°=right, 90°=forward, 180°=left, 270°=back)"
    echo "  Speeds: ${SPEEDS[*]} m/s"
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

readonly EXPERIMENTS=("$CHECKPOINT_DIR")

echo "========================================="
echo "Principled Spatial Control Pipeline"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Directions: ${DIRECTIONS[*]}°"
echo "Speeds: ${SPEEDS[*]} m/s"
echo "Total combinations: $((${#DIRECTIONS[@]} * ${#SPEEDS[@]}))"
echo "Python Command: $PYTHON_CMD"
echo "========================================="

###############################################################################
# Generation Mode
###############################################################################
generate_motions() {
    echo "Starting spatial control motion generation..."
    
    local total_combinations=$((${#EXPERIMENTS[@]} * ${#DIRECTIONS[@]} * ${#SPEEDS[@]}))
    local current=0
    
    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Processing experiment: $exp_name"
        
        for direction in "${DIRECTIONS[@]}"; do
            for speed in "${SPEEDS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Generating dir=${direction}°, speed=${speed}m/s for $exp_name"
                
                $PYTHON_CMD experiments/gen_PrincipledSpatialControl.py \
                    --expdir "$expdir" \
                    --target_dir "$direction" \
                    --target_speed "$speed" \
                    $GENERATION_PARAMS
                
                echo "✓ Completed generation for dir=${direction}°, speed=${speed}m/s"
            done
        done
        
        echo "✓ Completed all combinations for $exp_name"
    done
    
    echo "All spatial control motion generation completed!"
}

###############################################################################
# Rendering Mode  
###############################################################################
render_videos() {
    echo "Starting video rendering..."
    
    local total_combinations=$((${#EXPERIMENTS[@]} * ${#DIRECTIONS[@]} * ${#SPEEDS[@]}))
    local current=0
    
    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Rendering videos for experiment: $exp_name"
        
        for direction in "${DIRECTIONS[@]}"; do
            for speed in "${SPEEDS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Rendering dir=${direction}°, speed=${speed}m/s for $exp_name"
                
                # Format floating point numbers properly
                local formatted_dir="$direction.0"
                local formatted_speed="$speed"

                local pkl_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_speed-${formatted_speed}.pkl"

                if [ ! -f "$pkl_file" ]; then
                    echo "Warning: PKL file not found: $pkl_file"
                    echo "Skipping rendering for dir=${direction}°, speed=${speed}m/s"
                    continue
                fi
                
                $PYTHON_CMD primal/rendering/render.py "$pkl_file" false
                echo "✓ Completed rendering for dir=${direction}°, speed=${speed}m/s"
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
    
    local total_combinations=$((${#EXPERIMENTS[@]} * ${#DIRECTIONS[@]} * ${#SPEEDS[@]}))
    local current=0
    
    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        echo "Evaluating metrics for experiment: $exp_name"
        
        for direction in "${DIRECTIONS[@]}"; do
            for speed in "${SPEEDS[@]}"; do
                current=$((current + 1))
                echo "[$current/$total_combinations] Evaluating dir=${direction}°, speed=${speed}m/s for $exp_name"
                
                # Format floating point numbers properly
                local formatted_dir="$direction.0"
                local formatted_speed="$speed"

                local pkl_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_speed-${formatted_speed}.pkl"
                local output_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_speed-${formatted_speed}.txt"

                if [ ! -f "$pkl_file" ]; then
                    echo "Warning: PKL file not found: $pkl_file"
                    echo "Skipping evaluation for dir=${direction}°, speed=${speed}m/s"
                    continue
                fi
                
                echo "Evaluating: $pkl_file"
                echo "Output: $output_file"
                
                $PYTHON_CMD experiments/eval_generation.py "$pkl_file" > "$output_file"
                echo "✓ Completed evaluation for dir=${direction}°, speed=${speed}m/s"
            done
        done
        
        echo "✓ Completed all evaluations for $exp_name"
    done
    
    echo "All motion evaluation completed!"
    echo ""
    echo "Evaluation results saved in:"
    for expdir in "${EXPERIMENTS[@]}"; do
        local exp_name=$(basename "$expdir")
        for direction in "${DIRECTIONS[@]}"; do
            for speed in "${SPEEDS[@]}"; do
                local formatted_dir="$direction.0"
                local formatted_speed="$speed"
                local output_file="$OUTPUT_DIR/$exp_name/SFU_ema-True_reproj-True_inertial-True_dir-${formatted_dir}_speed-${formatted_speed}.txt"
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
echo "Directions processed: ${DIRECTIONS[*]}°"
echo "Speeds processed: ${SPEEDS[*]} m/s"
echo "========================================="