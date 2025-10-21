
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xavier_uniform_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 5).astype(np.float32)
    gain1 = 1.0
    input_dict1 = {"tensor": input1, "gain": gain1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 2, 3).astype(np.float64)
    gain2 = 1.0
    input_dict2 = {"tensor": input2, "gain": gain2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.zeros((3, 3), dtype=np.float32)
    gain3 = 1.0
    input_dict3 = {"tensor": input3, "gain": gain3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2).astype(np.float16)
    gain4 = 1.0
    input_dict4 = {"tensor": input4, "gain": gain4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.init.xavier_uniform_"] = xavier_uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.xavier_uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_uniform_'.")


check_valid('torch.nn.init.xavier_uniform_', generated_inputs['torch.nn.init.xavier_uniform_'], lib="torch", suffix=0)
