
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ones_like_inputs():
    list_of_inputs = []
    
    input1 = np.array([1, 2, 3])
    dtype1 = torch.float32
    requires_grad1 = False
    input_dict1 = {"input": input1, "dtype": dtype1, "requires_grad": requires_grad1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1, 2], [3, 4]])
    dtype2 = torch.float64
    requires_grad2 = True
    input_dict2 = {"input": input2, "dtype": dtype2, "requires_grad": requires_grad2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.zeros((2, 3, 4))
    dtype3 = torch.int32
    requires_grad3 = False
    input_dict3 = {"input": input3, "dtype": dtype3, "requires_grad": requires_grad3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-1, -2], [-3, -4]])
    dtype4 = torch.float16
    requires_grad4 = True
    input_dict4 = {"input": input4, "dtype": dtype4, "requires_grad": requires_grad4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0])
    dtype5 = torch.complex128
    requires_grad5 = False
    input_dict5 = {"input": input5, "dtype": dtype5, "requires_grad": requires_grad5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([])
    dtype6 = torch.float32
    requires_grad6 = True
    input_dict6 = {"input": input6, "dtype": dtype6, "requires_grad": requires_grad6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([1])
    dtype7 = torch.float64
    requires_grad7 = False
    input_dict7 = {"input": input7, "dtype": dtype7, "requires_grad": requires_grad7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1, 2, 3, 4], dtype=np.int64)
    dtype8 = torch.float32
    requires_grad8 = True
    input_dict8 = {"input": input8, "dtype": dtype8, "requires_grad": requires_grad8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(3, 3)
    dtype9 = torch.float16
    requires_grad9 = False
    input_dict9 = {"input": input9, "dtype": dtype9, "requires_grad": requires_grad9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs["torch.ones_like"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ones_like' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ones_like'.")


check_valid('torch.ones_like', generated_inputs['torch.ones_like'], lib="torch", suffix=0)
