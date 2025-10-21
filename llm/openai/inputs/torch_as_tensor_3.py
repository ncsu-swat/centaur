
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def as_tensor_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    dtype1 = torch.float32
    input_dict1 = {"data": input1, "dtype": dtype1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    dtype2 = torch.float64
    input_dict2 = {"data": input2, "dtype": dtype2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1, -2], [3, 4]])
    dtype3 = torch.int32
    input_dict3 = {"data": input3, "dtype": dtype3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 0.0, 0.0])
    dtype4 = torch.float16
    input_dict4 = {"data": input4, "dtype": dtype4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1, 2, 3, 4, 5])
    dtype5 = torch.int64
    input_dict5 = {"data": input5, "dtype": dtype5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    dtype6 = torch.float32
    input_dict6 = {"data": input6, "dtype": dtype6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([])
    dtype7 = torch.float32
    input_dict7 = {"data": input7, "dtype": dtype7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([1.5, 2.5, 3.5])
    dtype8 = torch.float64
    input_dict8 = {"data": input8, "dtype": dtype8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([[1, 0], [0, 1]])
    dtype9 = torch.bool
    input_dict9 = {"data": input9, "dtype": dtype9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    
    return list_of_inputs

generated_inputs["torch.as_tensor_3"] = as_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.as_tensor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.as_tensor_3'.")


check_valid('torch.as_tensor', generated_inputs['torch.as_tensor_3'], lib="torch", suffix=3)
