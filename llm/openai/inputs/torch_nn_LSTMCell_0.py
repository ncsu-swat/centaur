
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lstmcell_inputs():
    list_of_inputs = []

    input_size = 10
    hidden_size = 20

    input1 = np.random.rand(1, input_size).astype(np.float32)
    h_0_1 = np.random.rand(1, hidden_size).astype(np.float32)
    c_0_1 = np.random.rand(1, hidden_size).astype(np.float32)
    input_dict1 = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "bias": True,
        "input": input1,
        "h_0": h_0_1,
        "c_0": c_0_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.LSTMCell"] = lstmcell_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LSTMCell' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LSTMCell'.")


check_valid('torch.nn.LSTMCell', generated_inputs['torch.nn.LSTMCell'], lib="torch", suffix=0)
