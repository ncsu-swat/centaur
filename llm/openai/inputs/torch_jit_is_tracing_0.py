
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_tracing_inputs():
    list_of_inputs = []
    
    input1 = np.array([])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input1}))

    input2 = np.array([1, 2, 3])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input2}))

    input3 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input3}))
    
    input4 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input4}))

    input5 = np.array([-1, -2, -3])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input5}))

    input6 = np.array([1.0, 2.5, 3.7])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input6}))

    input7 = np.array([[1.1, 2.2], [3.3, 4.4]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input7}))

    input8 = np.array([0])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input8}))

    input9 = np.array([1, 0, 1])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input9}))

    input10 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input10}))

    input11 = np.array([1,2,3,4,5,6,7,8,9,10])
    list_of_inputs.append(copy.deepcopy({"torch.jit.is_tracing": input11}))

    return list_of_inputs

generated_inputs["torch.jit.is_tracing"] = is_tracing_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.is_tracing' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.is_tracing'.")


check_valid('torch.jit.is_tracing', generated_inputs['torch.jit.is_tracing'], lib="torch", suffix=0)
