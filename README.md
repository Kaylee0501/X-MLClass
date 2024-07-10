# X-MLClass: Open-world Multi-label Text Classification

This repository contains the code and dataset for the paper [Open-world Multi-label Text Classification with Extremely Weak Supervision](https://arxiv.org/abs/2407.05609).

We study open-world multi-label text classification under extremely weak supervision, where the user only provides a brief description for classification objectives without any labels or ground-truth label space.

## Installation
```bash
conda create -n X-MLClass python=3.9
conda activate X-MLClass
python -m pip install -r requirements.txt
```
If you need to use OpenAI APIs, you will need to obtain an API key [here](https://beta.openai.com/). 
```
export OPENAI_API_KEY=[your OpenAI API Key]
```

## Dataset
All datasets referenced in the paper are available [here](https://drive.google.com/drive/folders/1eX6awgaAVmRee2pnWb2bdaQGFP_zlmyS?usp=drive_link).


## Usage
There are three steps to follow the framework outlined in this paper.

1.  Initial label space construction.
2.  Assign labels using a custom keyphrase-chunk zero-shot textual entailment classifier.
3.  Label space improvement.

Below, we provide an example of open-world multi-label text classification using the AAPD dataset.

### 1. Label Space Construction

We placed the [keyphrases file](https://drive.google.com/file/d/1qN8RZlOrRcPxzCKZb0VuOyH_J8LQ5INg/view?usp=sharing) we generated in the dataset folder. You are also welcome to generate the keyphrases yourself using the following command:
```
CUDA_VISIBLE_DEVICES=... python llama_keyword.py \
    --path ./datasets \
    --data_dir train_texts_split_50.txt \
    --task AAPD \
    --batch_size 32 \
    --model meta-llama/Llama-2-13b-chat-hf \
    --output_dir llama_label2_50.txt
``` 

To generate the initial label space, use the following command. The output will be saved in `llama2/init_label_space.txt`
```
cd OpenWordMLTC/keyword_generator
bash label_space_construct.sh
```

### 2. Textual Entailment-based Classifier
```
cd OpenWordMLTC/zero-shot
bash multi_label_classifier.sh
```

Please note that in this code, we use the initial label space for multi-label text classification. The results presented in the paper are based on the label space after improvements made in the next step.

### 3.  Label Space Improvement
```
cd OpenWordMLTC/self_training
CUDA_VISIBLE_DEVICES=... python self_training.py \
    --path ../../datasets \
    --data_dir train_texts_split_50.txt \ 
    --keyphrase_dir llama2_label_50.txt\
    --task AAPD \
    --llama_model llama2 \
    --tail_set_size 500 \
    --majority_num 350 \
    --max_majority_num 5 \
    --sim_threshold 0.55 \
    --max_add_label 10 \
    --model MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33
```
The improved label space is stored in `llama2/result/update_labelspace.txt`