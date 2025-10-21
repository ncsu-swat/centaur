
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def xavier_normal_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 5).astype(np.float32)
    gain1 = 1.0
    input_dict1 = {"tensor": input1, "gain": gain1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(10, 2, 3).astype(np.float64)
    gain2 = 0.5
    input_dict2 = {"tensor": input2, "gain": gain2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 2).astype(np.float16)
    gain3 = 2.0
    input_dict3 = {"tensor": input3, "gain": gain3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.rand(3, 4, 5, 6).astype(np.float32)
    gain4 = 1.0
    input_dict4 = {"tensor": input4, "gain": gain4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.random.rand(7, 7).astype(np.float64)
    gain5 = 0.0
    input_dict5 = {"tensor": input5, "gain": gain5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.random.rand(1, 1, 1).astype(np.float32)
    gain6 = 1.5
    input_dict6 = {"tensor": input6, "gain": gain6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(4, 4).astype(np.float16)
    gain7 = 0.5
    input_dict7 = {"tensor": input7, "gain": gain7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(6, 6, 6).astype(np.float32)
    gain8 = 3.0
    input_dict8 = {"tensor": input8, "gain": gain8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.nn.init.xavier_normal_"] = xavier_normal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.xavier_normal_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.xavier_normal_'.")


check_valid('torch.nn.init.xavier_normal_', generated_inputs['torch.nn.init.xavier_normal_'], lib="torch", suffix=0)
