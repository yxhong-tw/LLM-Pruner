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
    --num_examples 512 \

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
    --num_examples 81 \

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
    --num_examples 384 \

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
    --num_examples 488 \

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
    --num_examples 256 \

# Compressed Llama-2-7b with Wikitext2 dataset and pruning ratio 0.4.
python3 hf_prune.py --pruning_ratio 0.4 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name wikitext2_0d4 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data wikitext2 \
    --num_examples 256 \

# Compressed Llama-2-7b with Wikitext2 dataset and pruning ratio 0.6.
python3 hf_prune.py --pruning_ratio 0.6 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name wikitext2_0d6 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data wikitext2 \
    --num_examples 256 \

# Compressed Llama-2-7b with Wikitext2 dataset and pruning ratio 0.8.
python3 hf_prune.py --pruning_ratio 0.8 \
    --block_wise \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --block_attention_layer_start 0 --block_attention_layer_end 0 \
    --pruner_type taylor \
    --device cpu  --eval_device cuda:2 \
    --save_ckpt_log_name wikitext2_0d8 \
    --base_model /workspace/models/Llama-2-7b-hf \
    --save_model \
    --seed 48763 \
    --iterative_steps 1 \
    --calibration_data wikitext2 \
    --num_examples 256 \
