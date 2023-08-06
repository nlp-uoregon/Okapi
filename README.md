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
You could get started chatting with the model by cloning our repository.
```
git clone https://github.com/nlp-uoregon/Okapi.git
cd Okapi
pip install -r requirements.txt
```

Then, you can try to chat with our model:
```python
from chat import pipeline

model_path = 'laiviet/okapi-vi-bloom'
p = pipeline(model_path, gpu=True)

instruction = 'Dịch câu sau sang Tiếng Việt' # Translate the following sentence into Vietnamese
prompt_input = 'The City of Eugene - a great city for the arts and outdoors. '

response = p.generate(instruction=instruction, prompt_input=prompt_input)
print(response)
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
