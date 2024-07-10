# train
CUDA_VISIBLE_DEVICES=0 python zero-shot-AAPD.py \
    --path ../../datasets \
    --data_dir train_texts_split_50.txt \
    --keyphrase_dir llama2_label_50.txt \
    --task AAPD \
    --dynamic_iter 3000 \
    --model MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33

# test
CUDA_VISIBLE_DEVICES=0 python zero-shot_AAPD_test.py \
    --path ../../datasets \
    --data_dir llama2/init_label_space.txt \
    --task AAPD \
    --model meta-llama/Llama-2-13b-chat-hf

CUDA_VISIBLE_DEVICES=0 python model_eval \
    --path ../../datasets \
    --data_dir llama2/test_performance \
    --keyphrase_dir llama2_label_test_50.txt \
    --task AAPD \
    --test_size 1000 \
    --output_dir llama2/test_performance/MLClass_result.txt