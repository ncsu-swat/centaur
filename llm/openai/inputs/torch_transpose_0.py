
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def transpose_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2, 3).numpy()
    dim0_1 = 0
    dim1_1 = 1
    input_dict1 = {"input": input1, "dim0": dim0_1, "dim1": dim1_1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(3, 2).numpy()
    dim0_2 = 0
    dim1_2 = 1
    input_dict2 = {"input": input2, "dim0": dim0_2, "dim1": dim1_2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(4, 5, 6).numpy()
    dim0_3 = 0
    dim1_3 = 2
    input_dict3 = {"input": input3, "dim0": dim0_3, "dim1": dim1_3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(5, 4, 3).numpy()
    dim0_4 = 1
    dim1_4 = 2
    input_dict4 = {"input": input4, "dim0": dim0_4, "dim1": dim1_4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 2, 2, 2).numpy()
    dim0_5 = 0
    dim1_5 = 3
    input_dict5 = {"input": input5, "dim0": dim0_5, "dim1": dim1_5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(3, 3, 3, 3).numpy()
    dim0_6 = 1
    dim1_6 = 2
    input_dict6 = {"input": input6, "dim0": dim0_6, "dim1": dim1_6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(2, 3, 4).numpy()
    dim0_7 = 0
    dim1_7 = 1
    input_dict7 = {"input": input7, "dim0": dim0_7, "dim1": dim1_7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(4, 3, 2).numpy()
    dim0_8 = 1
    dim1_8 = 2
    input_dict8 = {"input": input8, "dim0": dim0_8, "dim1": dim1_8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(2, 2).numpy()
    dim0_9 = 0
    dim1_9 = 1
    input_dict9 = {"input": input9, "dim0": dim0_9, "dim1": dim1_9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(5, 5).numpy()
    dim0_10 = 0
    dim1_10 = 1
    input_dict10 = {"input": input10, "dim0": dim0_10, "dim1": dim1_10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["torch.transpose"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.transpose'.")


check_valid('torch.transpose', generated_inputs['torch.transpose'], lib="torch", suffix=0)
