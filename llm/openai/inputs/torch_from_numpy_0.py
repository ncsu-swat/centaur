
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def from_numpy_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    list_of_inputs.append([input1])
    
    input2 = np.array([[1, 2], [3, 4]])
    list_of_inputs.append([input2])
    
    input3 = np.array([[-1, -2, -3], [0, 1, 2]])
    list_of_inputs.append([input3])
    
    input4 = np.array([1.0, 2.0, 3.0])
    list_of_inputs.append([input4])
    
    input5 = np.array([[1.1, 2.2], [3.3, 4.4]])
    list_of_inputs.append([input5])
    
    input6 = np.array([True, False, True])
    list_of_inputs.append([input6])
    
    input7 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    list_of_inputs.append([input7])
    
    input8 = np.array([])
    list_of_inputs.append([input8])
    
    input9 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append([input9])

    input10 = np.array([1, 2, 3], dtype=np.float32)
    list_of_inputs.append([input10])

    input11 = np.array([1,2,3], dtype=np.complex128)
    list_of_inputs.append([input11])
    
    return list_of_inputs

generated_inputs["torch.from_numpy"] = from_numpy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.from_numpy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.from_numpy'.")


check_valid('torch.from_numpy', generated_inputs['torch.from_numpy'], lib="torch", suffix=0)
