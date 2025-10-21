
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def multihead_attention_inputs():
    list_of_inputs = []
    
    input_dict1 = {
        'embed_dim': 32,
        'num_heads': 2,
        'dropout': 0.1,
        'bias': True,
        'add_bias_kv': False,
        'add_zero_attn': False,
        'kdim': 32,
        'vdim': 32,
        'batch_first': False,
        'dtype': np.float32,
        'query': np.random.rand(5, 32, 32).astype(np.float32),
        'key': np.random.rand(5, 32, 32).astype(np.float32),
        'value': np.random.rand(5, 32, 32).astype(np.float32),
        'key_padding_mask': np.random.randint(0, 2, (5, 32)).astype(np.bool_),
        'need_weights': True,
        'attn_mask': np.random.randint(0, 2, (5, 5)).astype(np.bool_),
        'average_attn_weights': True,
        'is_causal': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.MultiheadAttention"] = multihead_attention_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MultiheadAttention' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MultiheadAttention'.")


check_valid('torch.nn.MultiheadAttention', generated_inputs['torch.nn.MultiheadAttention'], lib="torch", suffix=0)
