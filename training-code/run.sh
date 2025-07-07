# Here is the script to run the training for Dimple models. 

# autoregressive alignment
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --main_process_port 28100 training.py \
    --seed 0 \
    --model Dimple-raw \
    --causal_attention yes \
    --max-pixel-scale 24 \
    --dataset llava_alignment \
    --block-attention no \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --warm_up 0.03 \
    --max-seq-length 1024 \
    --epoch 1 \
    --max-steps 2285 \
    --gradient-accumulation-steps 8 \
    --per-device-train-batch-size 8 \
    --mask_strategy single \
    --rescale_loss yes \
    --trainer-type autoregressive \
    --save-steps 1000 \
    --stage align \
    --mark autoreg-align \
    --save-ckpt

# autoregessive instruction tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --use_deepspeed --deepspeed_config_file deepspeed/stage-1.json  --main_process_port 28100 training.py \
    --seed 0 \
    --model Dimple-align \
    --causal_attention yes \
    --max-pixel-scale 24 \
    --dataset llava_next \
    --block-attention no \
    --learning-rate 2e-5 \
    --batch-size 128 \
    --warm_up 0.03 \
    --max-seq-length 1024 \
    --epoch 1 \
    --max-steps 5690 \
    --gradient-accumulation-steps 16 \
    --per-device-train-batch-size 2 \
    --mask_strategy single \
    --rescale_loss yes \
    --trainer-type autoregressive \
    --save-steps 1000 \
    --stage instruct \
    --mark autoreg-instruct \
    --save-ckpt

# dlm instruct
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --use_deepspeed --deepspeed_config_file deepspeed/stage-1.json  --main_process_port 28100 training.py \
    --seed 0 \
    --model Dimple-ar-instruct \
    --causal_attention no \
    --max-pixel-scale 24 \
    --dataset llava_next \
    --block-attention yes \
    --learning-rate 5e-6 \
    --max-grad-norm 0.01 \
    --batch-size 128 \
    --warm_up 0.03 \
    --max-seq-length 1024 \
    --epoch 1 \
    --max-steps 5690 \
    --gradient-accumulation-steps 32 \
    --per-device-train-batch-size 1 \
    --mask_strategy singple \
    --rescale_loss yes \
    --trainer-type dlm \
    --save-steps 1000 \
    --stage diffusion-recovery \
    --mark diffusion-recovery \
    --save-ckpt \
    --use-shard