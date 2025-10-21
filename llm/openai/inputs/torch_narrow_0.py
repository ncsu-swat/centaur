
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def narrow_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict1 = {"input": input1, "dim": 0, "start": 0, "length": 2}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict2 = {"input": input2, "dim": 1, "start": 1, "length": 2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).numpy()
    input_dict3 = {"input": input3, "dim": -1, "start": -1, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 3, 4).numpy()
    input_dict4 = {"input": input4, "dim": 0, "start": 1, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict5 = {"input": input5, "dim": 1, "start": 2, "length": 3}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict6 = {"input": input6, "dim": 0, "start": 1, "length": 3}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([[1, 2], [3, 4]]).numpy()
    input_dict7 = {"input": input7, "dim": -2, "start": 0, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(3, 4, 5, 6).numpy()
    input_dict8 = {"input": input8, "dim": 2, "start": 1, "length": 2}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.tensor([[1, 2, 3, 4]]).numpy()
    input_dict9 = {"input": input9, "dim": 1, "start": 0, "length": 4}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randint(0, 10, (4, 4)).numpy()
    input_dict10 = {"input": input10, "dim": 0, "start": 3, "length": 1}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.narrow"] = narrow_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.narrow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.narrow'.")


check_valid('torch.narrow', generated_inputs['torch.narrow'], lib="torch", suffix=0)
