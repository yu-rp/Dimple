
# First, setup the environment for both lmms-eval and dimple.
# Second, run the following commands to evaluate dimple on various tasks.
cd lmms-eval

# gqa,vizwiz_vqa_test,scienceqa_img,mmbench_en_test,pope,mathvista_testmini,ai2d,chartqa,mme,ocrbench
OPENAI_API_KEY=MY_OPENAI_API_KEY python3 -m accelerate.commands.launch \
    -m lmms_eval \
    --model dimple \
    --model_args causal_attention=False,max_pixel_scale=-1,interleave_visuals=False \
    --gen_kwargs max_new_tokens=8,steps=8,temperature=0.0,top_p=0.01,alg=maskgit_plus,alg_temp=0.0,use_cache=True,alg_p_threshold=0.0,use_original_confidence=False,decoding_pipeline=dream \
    --tasks gqa,vizwiz_vqa_test,scienceqa_img,mmbench_en_test,pope,mathvista_testmini,ai2d,chartqa,mme,ocrbench \
    --batch_size 1 \
    --log_samples  \
    --log_samples_suffix dimple \
    --output_path ./logs/ \
    --limit 1024

# mmmu_val
OPENAI_API_KEY=MY_OPENAI_API_KEY python3 -m accelerate.commands.launch \
    -m lmms_eval \
    --model dimple \
    --model_args causal_attention=False,max_pixel_scale=-1,interleave_visuals=True \
    --gen_kwargs max_new_tokens=8,steps=8,temperature=0.0,top_p=0.01,alg=maskgit_plus,alg_temp=0.0,use_cache=True,alg_p_threshold=0.0,use_original_confidence=False,decoding_pipeline=dream \
    --tasks mmmu_val \
    --batch_size 1 \
    --log_samples  \
    --log_samples_suffix dimple \
    --output_path ./logs/ \
    --limit 1024

# mmvet
OPENAI_API_KEY=MY_OPENAI_API_KEY python3 -m accelerate.commands.launch \
    -m lmms_eval \
    --model dimple \
    --model_args causal_attention=False,max_pixel_scale=-1,interleave_visuals=True \
    --gen_kwargs max_new_tokens=64,steps=64,temperature=0.0,top_p=0.01,alg=maskgit_plus,alg_temp=0.0,use_cache=True,alg_p_threshold=0.0,use_original_confidence=False,decoding_pipeline=dream \
    --tasks mmvet \
    --batch_size 1 \
    --log_samples  \
    --log_samples_suffix dimple \
    --output_path ./logs/ \
    --limit 1024