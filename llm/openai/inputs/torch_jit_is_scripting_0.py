
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_scripting_inputs():
    list_of_inputs = []
    
    input1 = np.array([])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input1}))
    
    input2 = np.array([1, 2, 3])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input2}))
    
    input3 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input3}))
    
    input4 = np.array([[-1, 0, 1], [2, -2, 0]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input4}))
    
    input5 = np.array([1.0, 2.5, 3.7])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input5}))
    
    input6 = np.array([[1.1, 2.2], [3.3, 4.4]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input6}))
    
    input7 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input7}))
    
    input8 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input8}))
    
    input9 = np.array([True, False, True, False])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input9}))

    input10 = np.array([1, 2, 3], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input10}))
    
    input11 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_scripting": input11}))

    return list_of_inputs

generated_inputs["torch.jit.is_scripting"] = is_scripting_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.is_scripting' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.is_scripting'.")


check_valid('torch.jit.is_scripting', generated_inputs['torch.jit.is_scripting'], lib="torch", suffix=0)
