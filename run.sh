#!/bin/bash

# Compressed Llama-2-7b with C4 dataset and pruning ratio 0.2.
python3 hf_prune.py --pruning_ratio 0.2 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name c4_0d2 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data c4 \

# Compressed Llama-2-7b with Alpaca dataset and pruning ratio 0.2.
python3 hf_prune.py --pruning_ratio 0.2 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name alpaca_0d2 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data alpaca \

# Compressed Llama-2-7b with OpenBookQA dataset and pruning ratio 0.2.
python3 hf_prune.py --pruning_ratio 0.2 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name openbookqa_0d2 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data openbookqa \

# Compressed Llama-2-7b with PIQA dataset and pruning ratio 0.2.
python3 hf_prune.py --pruning_ratio 0.2 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name piqa_0d2 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data piqa \

# Compressed Llama-2-7b with Wikitext2 dataset and pruning ratio 0.2.
python3 hf_prune.py --pruning_ratio 0.2 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name wikitext2_0d2 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data wikitext2 \
