#! /bin/bash

export MASTER_PORT=$(((RANDOM % 1000) + 10000))
export MASTER_ADDR=0
export NODE_RANK=0
export OUTPUT_DIR="output_dir"
export IMAGENET_PATH="/media/llliutc/Elements/data/imagenet/ILSVRC/Data/CLS-LOC"

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
#  -m debugpy --listen 5000 --wait-for-client main_fractalgen.py \
# --model fractalar_in64 --img_size 64 --num_conds 1 \
# --batch_size 2 --eval_freq 40 --save_last_freq 10 \
# --epochs 5 --warmup_epochs 1 \
# --blr 5.0e-5 --weight_decay 0.05 --attn_dropout 0.1 --proj_dropout 0.1 --lr_schedule cosine \
# --gen_bsz 2 --num_images 8000 --num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
# --output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
# --data_path ${IMAGENET_PATH} --grad_checkpointing --online_eval

# unconditional
# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
#  -m debugpy --listen 5000 --wait-for-client main_fractalgen.py \
# --model fractalar_in64 --img_size 64 --num_conds 1 \
# --nll_bsz 16 --nll_forward_number 1 \
# --output_dir pretrained_models/fractalar_in64 \
# --resume pretrained_models/fractalar_in64 \
# --data_path ${IMAGENET_PATH} --seed 0 --evaluate_nll

# class-conditional generation
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
 -m debugpy --listen 5000 --wait-for-client main_fractalgen.py \
--model fractalar_in64 --img_size 64 --num_conds 1 \
--gen_bsz 2 --num_images 50000 \
--num_iter_list 64,16 --cfg 11.0 --cfg_schedule linear --temperature 1.03 \
--output_dir pretrained_models/fractalar_in64 \
--resume pretrained_models/fractalar_in64 \
--data_path ${IMAGENET_PATH} --seed 0 --evaluate_gen