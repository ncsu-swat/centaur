
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def transformer_decoder_layer_inputs():
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
        'tgt': np.random.rand(32, 32, 512).astype(np.float32),
        'memory': np.random.rand(10, 32, 512).astype(np.float32),
        'tgt_mask': np.random.randint(0, 2, size=(32, 32)).astype(np.float32),
        'memory_mask': np.random.randint(0, 2, size=(10, 10)).astype(np.float32),
        'tgt_key_padding_mask': np.random.randint(0, 2, size=(32, 32)).astype(np.float32),
        'memory_key_padding_mask': np.random.randint(0, 2, size=(10, 32)).astype(np.float32),
        'tgt_is_causal': False,
        'memory_is_causal': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["torch.nn.TransformerDecoderLayer"] = transformer_decoder_layer_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.TransformerDecoderLayer' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TransformerDecoderLayer'.")


check_valid('torch.nn.TransformerDecoderLayer', generated_inputs['torch.nn.TransformerDecoderLayer'], lib="torch", suffix=0)
