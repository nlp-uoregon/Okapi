import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import Prompter

class pipeline(object):

    def __init__(self, model_path=None, gpu=False):
        self.model_path = model_path
        self.use_gpu = gpu
        self.prompter = Prompter()
        
        print("Loading model's weights ...")
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16)
        
        if self.use_gpu:
            self.model.cuda()

    def to_cuda(self, inputs):
        return {k: v.cuda() for k, v in inputs.items()}

    def generate(self, instruction, prompt_input=None):
        prompt = self.prompter.generate_prompt(instruction, prompt_input)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        if self.use_gpu:
            inputs = self.to_cuda(inputs)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.75,
            top_k=40
        )
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.prompter.get_response(output)
        return response
