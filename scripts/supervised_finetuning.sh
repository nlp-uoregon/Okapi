lang=$1
model_path=bloom-7b1
sft_output_dir=ckpts/sft_models/sft_model_$lang
data_path=dataset/alpaca-chatgpt/$lang.json

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" torchrun --nproc_per_node=8 \
        supervised_training.py \
        --model_path=$model_path \
        --data_path=$data_path \
        --output_dir=$sft_output_dir