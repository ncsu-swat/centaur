
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def transformer_encoder_layer_inputs():
    list_of_inputs = []
    
    input_dict_1 = {
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'layer_norm_eps': 1e-05,
        'batch_first': False,
        'norm_first': False,
        'bias': True,
        'dtype': np.float32,
        'src': np.random.rand(10, 32, 512).astype(np.float32),
        'src_mask': None,
        'src_key_padding_mask': None,
        'is_causal': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        'd_model': 256,
        'nhead': 4,
        'dim_feedforward': 1024,
        'dropout': 0.2,
        'activation': 'gelu',
        'layer_norm_eps': 1e-06,
        'batch_first': True,
        'norm_first': True,
        'bias': False,
        'dtype': np.float64,
        'src': np.random.rand(32, 10, 256).astype(np.float64),
        'src_mask': np.random.randint(0, 2, size=(32, 10)).astype(np.bool_),
        'src_key_padding_mask': np.random.randint(0, 2, size=(32, 10)).astype(np.bool_),
        'is_causal': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    
    return list_of_inputs

generated_inputs["torch.nn.TransformerEncoderLayer"] = transformer_encoder_layer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.TransformerEncoderLayer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TransformerEncoderLayer'.")


check_valid('torch.nn.TransformerEncoderLayer', generated_inputs['torch.nn.TransformerEncoderLayer'], lib="torch", suffix=0)
