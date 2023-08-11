lang=$1
sft_model_path=sft_models/sft_model_$lang
data_path=datasets/multilingual-ranking-data-42k/$lang.json
rm_output_dir=ckpts/reward_models/rm_$lang

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" torchrun --nproc_per_node=8 \
                    reward_modeling.py \
                    --model_path=$sft_model_path \
                    --tokenizer_path=$sft_model_path \
                    --data_path=$data_path \
                    --output_dir=$rm_output_dir \
