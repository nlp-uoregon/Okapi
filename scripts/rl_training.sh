lang=$1
sft_model_path=sft_models/sft_model_$lang
data_path=dataset/elk-chatgpt/$lang.json
rm_output_dir=ckpts/reward_models/rm_$lang
rl_output_dir=ckpts/rlhf/okapi-$lang

accelerate launch --config_file configs/rlhf_config.yaml \
    rl_training.py \
    --tokenizer_path=$sft_model_path \
    --sft_model_path=$sft_model_path \
    --reward_model_path=$rm_output_dir \
    --reward_tokenizer_path=$rm_output_dir \
    --data_path=$data_path \
    --output_dir=$rl_output_dir