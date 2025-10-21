
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def gru_inputs():
    list_of_inputs = []

    input_dict_1 = {
        'input_size': 10,
        'hidden_size': 20,
        'num_layers': 1,
        'bias': True,
        'batch_first': False,
        'dropout': 0.0,
        'bidirectional': False,
        'dtype': np.float32,
        'input': np.random.rand(5, 3, 10).astype(np.float32),
        'h_0': np.random.rand(1, 3, 20).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["torch.nn.GRU"] = gru_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GRU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GRU'.")


check_valid('torch.nn.GRU', generated_inputs['torch.nn.GRU'], lib="torch", suffix=0)
