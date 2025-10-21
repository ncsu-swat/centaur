
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flip_inputs():
    list_of_inputs = []
    
    input1 = torch.arange(8).view(2, 2, 2).numpy()
    dims1 = [0, 1]
    
    input2 = torch.randn(3, 4, 5).numpy()
    dims2 = [1]
    
    input3 = torch.ones(2, 3, 4, 5).numpy()
    dims3 = [0, 2]
    
    input4 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    dims4 = [0]
    
    input5 = torch.randn(2, 2).numpy()
    dims5 = [0, 1]
    
    input6 = torch.randint(0, 10, (5,)).numpy()
    dims6 = [0]
    
    input7 = torch.randn(1, 3, 3).numpy()
    dims7 = [1, 2]

    input8 = torch.zeros(4, 4, 4, 4).numpy()
    dims8 = [3]
    
    input9 = torch.arange(12).view(3, 4).numpy()
    dims9 = [0]
    
    input10 = torch.randn(2, 3, 4).numpy()
    dims10 = [2]
    
    input_dict1 = {"input": input1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input_dict2 = {"input": input2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {"input": input3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {"input": input4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {"input": input5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input_dict6 = {"input": input6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {"input": input7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {"input": input8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input_dict9 = {"input": input9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input_dict10 = {"input": input10, "dims": dims10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.flip"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flip'.")


check_valid('torch.flip', generated_inputs['torch.flip'], lib="torch", suffix=0)
