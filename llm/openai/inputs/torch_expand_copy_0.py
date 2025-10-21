
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def expand_copy_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    size1 = (3,)
    
    input2 = np.array([[1, 2], [3, 4]])
    size2 = (2, 2)
    
    input3 = np.array([1.0, 2.0, 3.0])
    size3 = (1, 3)
    
    input4 = np.array([[-1, 0, 1]])
    size4 = (1, 3)
    
    input5 = np.array([1, 2, 3, 4, 5])
    size5 = (5,)
    
    input6 = np.array([1, 2])
    size6 = (2,)
    
    input7 = np.array([[1.1, 2.2], [3.3, 4.4]])
    size7 = (2, 2)
    
    input_dict1 = {"self": input1, "size": size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input_dict2 = {"self": input2, "size": size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {"self": input3, "size": size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {"self": input4, "size": size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {"self": input5, "size": size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input_dict6 = {"self": input6, "size": size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input_dict7 = {"self": input7, "size": size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.expand_copy"] = expand_copy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.expand_copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.expand_copy'.")


check_valid('torch.expand_copy', generated_inputs['torch.expand_copy'], lib="torch", suffix=0)
