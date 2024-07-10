CUDA_VISIBLE_DEVICES=0 python dynamic_GMM.py \
    --path ../../datasets \
    --keyphrase_dir llama2_label_50.txt \
    --task AAPD \
    --dynamic_iter 3000 \
    --cluster_size 84 \
    --model meta-llama/Llama-2-13b-chat-hf \
    --batch_size 2 \
    --output_file init_labelspace.txt

CUDA_VISIBLE_DEVICES=0 python get_init_labelspace.py \
    --path ../../datasets \
    --data_dir llama2/init_labelspace.txt \
    --task AAPD \
    --lower_bound 0.80 \
    --model meta-llama/Llama-2-13b-chat-hf \
    --output_dir llama2/init_label_space.txt