<div align="center">

# Native Hybrid Attention for Efficient Sequence Modeling
[![arXiv](https://img.shields.io/badge/Arxiv-2510.07019-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.07019)
[![stars](https://img.shields.io/github/stars/JusenD/NHA)](https://github.com/JusenD/NHA/stargazers)

</div>

🎉 Welcome to NHA! This repository provides the implementation of [Native Hybrid Attention for Efficient Sequence Modeling](https://arxiv.org/abs/2510.07019). We include scripts for training, hybridization with pretrained LLMs, and evaluation.

## 📦 Installation

Create a new virtual environment and install the package from source:
```bash
conda create -n nha python=3.10
conda activate nha

pip install -e .
```
This will install all required dependencies.

## 🚀 Train

### 1. Prepare Datasets
Please follow the instructions [here](training/README.md) to prepare the datasets.

### 2. Model Configuration
Modify the configuration file (e.g., [nha_340M.json](training/configs/nha_340M.json)) to adjust the model architecture.

### 3. Run Training
Set the `model` argument to the configuration file and run:
```bash
cd training

bash train.sh \
  node=2 \
  gpus=8 \
  type=nha \
  lr=3e-4 \
  steps=30720 \
  batch=16 \
  update=1 \
  logging=16 \
  warmup=1024 \
  context=2048 \
  path=SlimPajama/nha-15B \
  project=SlimPajama \
  model=configs/nha_340M.json \
  tokenizer=fla-hub/gla-1.3B-100B \
  data=SlimPajama-627B \
  cache=data/chunk1/train
```

## 🪄 Hybridization with Pretrained LLMs

### 1. Initialize the NHA Model
Prepare the model you used (e.g., Llama3-8B).

Modify the config file and set the `model_type` to `llama_nha` to ensure NHA is loaded with the pretrained weights.

Then run the script below to initialize the gate projection using the key projection weights:
```bash
python convert_weights.py --input_path /path/to/your/input/model --output_path /path/to/your/output/model
```

### 2. Finetune the Model
Data preparation is the same as above. For hybridization training, we **freeze the FFN layers**:

```bash
cd training

bash train.sh \
  nodes=4 \
  gpus=8 \
  finetune=1 \
  froze_ffn=1 \
  type=llama_nha \
  lr=3e-5 \
  steps=10240 \
  batch=2 \
  update=4 \
  context=2048 \
  path=SlimPajama-finetune/llama-nha \
  project=SlimPajama-finetune \
  model=/path/to/your/output/model \
  data=Slimpajama \
  cache=data/llama3/train
```

## 📊 Evaluation
To evaluate on commonsense reasoning benchmarks:
```bash
MODEL_PATH=training/SlimPajama/nha-15B/checkpoint-30720

accelerate launch --multi_gpu evals/harness.py --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16 \
    --tasks arc_easy,arc_challenge,hellaswag,lambada_standard,piqa,winogrande,wikitext \
    --output_path eval_results \
    --batch_size 32 \
    --device cuda
```

For recall-intensive tasks, we recommend using the [prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention) repository.

## ⭐ Acknowledgements

This repository is built upon [flash-linear-attention](https://github.com/fla-org/flash-linear-attention). The evaluation is supported by [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [prefix-linear-attention](https://github.com/HazyResearch/prefix-linear-attention). Thank sincerely for their contribution!