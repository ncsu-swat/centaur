
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def uniform_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a1 = 0.0
    b1 = 1.0
    input_dict1 = {"tensor": input1, "a": a1, "b": b1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(2, 3).astype(np.float32)
    a2 = -1.0
    b2 = 1.0
    input_dict2 = {"tensor": input2, "a": a2, "b": b2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.zeros((5, 4, 3), dtype=np.float32)
    a3 = 5.0
    b3 = 10.0
    input_dict3 = {"tensor": input3, "a": a3, "b": b3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.ones((2, 2), dtype=np.float32)
    a4 = -0.5
    b4 = 0.5
    input_dict4 = {"tensor": input4, "a": a4, "b": b4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0], dtype=np.float32)
    a5 = 2.0
    b5 = 3.0
    input_dict5 = {"tensor": input5, "a": a5, "b": b5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.init.uniform_"] = uniform_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.uniform_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.uniform_'.")


check_valid('torch.nn.init.uniform_', generated_inputs['torch.nn.init.uniform_'], lib="torch", suffix=0)
