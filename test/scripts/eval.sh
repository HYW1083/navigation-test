#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="/home/lunet/cohw2/Projects/Test/Qwen3-VL/Qwen3-VL-8B-Instruct" 
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="evaluation"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT eval.py --model_path $CHECKPOINT --habitat_config_path $CONFIG --output_path $OUTPUT_PATH

