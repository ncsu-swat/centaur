
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lstm_inputs():
    list_of_inputs = []
    input_size = 10
    hidden_size = 20
    num_layers = 1
    bias = True
    batch_first = False
    dropout = 0.0
    bidirectional = False
    proj_size = 0
    dtype = np.float32

    input_1 = np.random.rand(5, 3, input_size).astype(dtype)
    h_0_1 = torch.zeros((num_layers, 3, hidden_size), dtype=torch.float32).numpy()
    c_0_1 = torch.zeros((num_layers, 3, hidden_size), dtype=torch.float32).numpy()

    input_dict_1 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bias": bias,
        "batch_first": batch_first,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "proj_size": proj_size,
        "dtype": dtype,
        "input": input_1,
        "h_0": h_0_1,
        "c_0": c_0_1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    return list_of_inputs

generated_inputs["torch.nn.LSTM"] = lstm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LSTM' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTM'.")


check_valid('torch.nn.LSTM', generated_inputs['torch.nn.LSTM'], lib="torch", suffix=0)
