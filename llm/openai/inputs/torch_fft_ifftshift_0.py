
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ifftshift_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, 2.0, 3.0, 4.0]).numpy()
    dim1 = 0
    input_dict1 = {"input": input1, "dim": dim1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 2).numpy()
    dim2 = 1
    input_dict2 = {"input": input2, "dim": dim2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(3, 4, 5).numpy()
    dim3 = (0, 1)
    input_dict3 = {"input": input3, "dim": dim3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 3, 4, 5).numpy()
    dim4 = (1, 3)
    input_dict4 = {"input": input4, "dim": dim4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(5).numpy()
    dim5 = None
    input_dict5 = {"input": input5, "dim": dim5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2, 2, 2).numpy()
    dim6 = 0
    input_dict6 = {"input": input6, "dim": dim6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(4, 4).numpy()
    dim7 = 1
    input_dict7 = {"input": input7, "dim": dim7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randn(2, 3).numpy()
    dim8 = (0)
    input_dict8 = {"input": input8, "dim": dim8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(5, 5, 5).numpy()
    dim9 = (0, 2)
    input_dict9 = {"input": input9, "dim": dim9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(10, 10, 10, 10).numpy()
    dim10 = (1, 3)
    input_dict10 = {"input": input10, "dim": dim10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    

    return list_of_inputs

generated_inputs["torch.fft.ifftshift"] = ifftshift_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.fft.ifftshift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.fft.ifftshift'.")


check_valid('torch.fft.ifftshift', generated_inputs['torch.fft.ifftshift'], lib="torch", suffix=0)
