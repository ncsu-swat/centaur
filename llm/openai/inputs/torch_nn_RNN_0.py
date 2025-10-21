
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rnn_inputs():
    list_of_inputs = []
    input_size = 10
    hidden_size = 20
    num_layers = 2
    nonlinearity = 'tanh'
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    dtype = np.float32

    input1 = np.random.rand(5, 3, input_size).astype(dtype)
    hx1 = np.zeros((num_layers, 3, hidden_size), dtype=dtype)
    input_dict1 = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'nonlinearity': nonlinearity,
        'bias': bias,
        'batch_first': batch_first,
        'dropout': dropout,
        'bidirectional': bidirectional,
        'dtype': dtype,
        'input': input1,
        'hx': hx1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.RNN"] = rnn_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.RNN' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.RNN'.")


check_valid('torch.nn.RNN', generated_inputs['torch.nn.RNN'], lib="torch", suffix=0)
