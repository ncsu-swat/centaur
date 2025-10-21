
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_isin_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[1, 2], [3, 4]]).numpy()
    test_elements1 = torch.tensor([2, 3]).numpy()
    assume_unique1 = True
    invert1 = False
    input_dict1 = {
        "elements": input1,
        "test_elements": test_elements1,
        "assume_unique": assume_unique1,
        "invert": invert1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    test_elements2 = torch.tensor([3, 5, 7]).numpy()
    assume_unique2 = False
    invert2 = False
    input_dict2 = {
        "elements": input2,
        "test_elements": test_elements2,
        "assume_unique": assume_unique2,
        "invert": invert2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    test_elements3 = torch.tensor([1, 4]).numpy()
    assume_unique3 = True
    invert3 = True
    input_dict3 = {
        "elements": input3,
        "test_elements": test_elements3,
        "assume_unique": assume_unique3,
        "invert": invert3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-1, -2, -3, -4]).numpy()
    test_elements4 = torch.tensor([-2, -4, 0]).numpy()
    assume_unique4 = False
    invert4 = False
    input_dict4 = {
        "elements": input4,
        "test_elements": test_elements4,
        "assume_unique": assume_unique4,
        "invert": invert4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    test_elements5 = torch.tensor([2.0, 4.0]).numpy()
    assume_unique5 = True
    invert5 = False
    input_dict5 = {
        "elements": input5,
        "test_elements": test_elements5,
        "assume_unique": assume_unique5,
        "invert": invert5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1, 2, 3]).numpy()
    test_elements6 = torch.tensor([1, 2, 3]).numpy()
    assume_unique6 = True
    invert6 = True
    input_dict6 = {
        "elements": input6,
        "test_elements": test_elements6,
        "assume_unique": assume_unique6,
        "invert": invert6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([1, 1, 2, 2, 3]).numpy()
    test_elements7 = torch.tensor([1, 2]).numpy()
    assume_unique7 = False
    invert7 = False
    input_dict7 = {
        "elements": input7,
        "test_elements": test_elements7,
        "assume_unique": assume_unique7,
        "invert": invert7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([1]).numpy()
    test_elements8 = torch.tensor([1,2,3]).numpy()
    assume_unique8 = True
    invert8 = False
    input_dict8 = {
        "elements": input8,
        "test_elements": test_elements8,
        "assume_unique": assume_unique8,
        "invert": invert8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1, 2, 3]).numpy()
    test_elements9 = torch.tensor([1]).numpy()
    assume_unique9 = True
    invert9 = False
    input_dict9 = {
        "elements": input9,
        "test_elements": test_elements9,
        "assume_unique": assume_unique9,
        "invert": invert9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([1, 2, 3]).numpy()
    test_elements10 = torch.tensor([4, 5, 6]).numpy()
    assume_unique10 = False
    invert10 = True
    input_dict10 = {
        "elements": input10,
        "test_elements": test_elements10,
        "assume_unique": assume_unique10,
        "invert": invert10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.isin"] = torch_isin_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isin'.")


check_valid('torch.isin', generated_inputs['torch.isin'], lib="torch", suffix=0)
