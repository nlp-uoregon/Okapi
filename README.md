<h1 align="center"> <p> Okapi </p></h1>
<h3 align="center">
    <p>Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback</p>
</h3>

## Overview

**Okapi dataset** consists of instruction resources for 26 different languages, including ChatGPT prompts, instruction datasets and response ranking data and benchmark datasets.

**Okapi models** are a series of RLHF-based instruction-tuned LLMs for multiple languages on Okapi dataset.

## Dataset Creation

we perform a data collection process to prepare necessary data for our multilingual framework Okapi in four major steps:

1. **English instruction generation**: Firstly, we obtain 52K instructions for tuning LLMs from Alpaca. Then, we apply the Self-Instruct procedure as Alpaca to generate more 106K additional instructions to a larger dataset for our RLHF-based models.
2. **Instruction Translation**: Given the 158K English instructions from Alpaca
and our generation process, we utilize ChatGPT to translate them into 26 target languages
3. **Ranking Data Production**: We then employ ChatGPT to rank the response outputs.
4. **Evaluation data creation**: We employ three datasets in Eleuther AI Language Model Evaluation Harness Leaderboard : AI2 Reasoning Challenge (ARC), HellaSwag, and MMLU, to evaluate the model performance by translating them into 26 selected languages using ChatGPT. Checkout [here](https://github.com/laiviet/lm-evaluation-harness)

## Model
Using our Okapi dataset dataset and the RLHF-based technique, we have developed a diverse range of language models for 26 seletec langauges, built upon LLaMA and BLOOM. You can access these models through huggingface. Our instruction-tuned multilingual Okapi models are available [here](https://huggingface.co/laiviet). 


## Usage
You could get started chatting with the model by using the *transformers* library. We suggest you should install transformers with this version:
```
pip install transformers=4.29.2
```

Notes that to chat with our model, you also need the `prompter.py` file inside the `utils` folder.
```python
from utils import Prompter
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = 'path/to/model_weight'
model_path = 'laiviet/okapi-vi-bloom'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16).cuda()

prompter = Prompter()
instruction = 'Give three tips for staying healthy.'
prompt_input = ''

prompt = prompter.generate_prompt(instruction, prompt_input)
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
output = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    top_p=0.75,
    top_k=40
)
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(prompter.get_response(output))
```

## Training

### Set up the environment:
```bash
conda create -n okapi python=3.9
conda activate okapi
pip install -r requirements.txt
```

### Training with multiple GPUs:

1. Supervised Fine-tuning
```bash
bash scripts/supervised_finetuning.sh [LANG]
```

2. Reward Modeling
```bash
bash scripts/reward_modeling.sh [LANG]
```

3. Finetuning with RLHF
```bash
bash scripts/rl_training.sh [LANG]
```

## Citation
If you use the data, model or code in this repository, please cite:

```
@article{dac2023okapi,
  title={Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author={Dac Lai, Viet and Van Nguyen, Chien and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A and Nguyen, Thien Huu},
  journal={arXiv e-prints},
  pages={arXiv--2307},
  year={2023}
}
```