import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc

model_path = "opt-1.3b"

def create_model():
    model = AutoModelForCausalLM.from_pretrained('model_path', device_map="auto")
    print(model)
    return model

class ModelConstructor:
    def __init__(self, model_path = "opt-1.3b"):
        self.model_path = model_path

    def create_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        return model
