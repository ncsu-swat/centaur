
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def swapdims_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]).numpy()
    input_dict1 = {"input": input1, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[[0,1],[2,3]],[[4,5],[6,7]]]).numpy()
    input_dict2 = {"input": input2, "dim0": 0, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {"input": input3, "dim0": 1, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randint(0, 10, (5,)).numpy()
    input_dict4 = {"input": input4, "dim0": 0, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.zeros(2, 2).numpy()
    input_dict5 = {"input": input5, "dim0": -1, "dim1": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.ones(3, 4, 5).numpy()
    input_dict6 = {"input": input6, "dim0": -2, "dim1": -1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 4, 5).numpy()
    input_dict7 = {"input": input7, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.randint(0, 10, (4, 4)).numpy()
    input_dict8 = {"input": input8, "dim0": -1, "dim1": -2}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(1, 2, 3).numpy()
    input_dict9 = {"input": input9, "dim0": 0, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.zeros(2, 1, 3, 4).numpy()
    input_dict10 = {"input": input10, "dim0": 1, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.swapdims"] = swapdims_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.swapdims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.swapdims'.")


check_valid('torch.swapdims', generated_inputs['torch.swapdims'], lib="torch", suffix=0)
