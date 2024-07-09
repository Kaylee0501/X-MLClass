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
