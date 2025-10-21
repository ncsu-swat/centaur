
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def l1loss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(2, 3).astype(np.float32)
    target1 = np.random.rand(2, 3).astype(np.float32)
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, 4, 4).astype(np.float32)
    target2 = np.random.rand(4, 4, 4).astype(np.float32)
    input_dict2 = {
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 5).astype(np.float32)
    target3 = np.random.rand(1, 5).astype(np.float32)
    input_dict3 = {
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2).astype(np.float64)
    target4 = np.random.rand(2, 2).astype(np.float64)
    input_dict4 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 3, 3).astype(np.float32)
    target5 = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict5 = {
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(5, 2).astype(np.float32)
    target6 = np.random.rand(5, 2).astype(np.float32)
    input_dict6 = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2).astype(np.float32)
    target7 = np.random.rand(2).astype(np.float32)
    input_dict7 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(1, 1, 1).astype(np.float32)
    target8 = np.random.rand(1, 1, 1).astype(np.float32)
    input_dict8 = {
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(4, 1).astype(np.float32)
    target9 = np.random.rand(4, 1).astype(np.float32)
    input_dict9 = {
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 3, 1).astype(np.float32)
    target10 = np.random.rand(2, 3, 1).astype(np.float32)
    input_dict10 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.L1Loss"] = l1loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.L1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.L1Loss'.")


check_valid('torch.nn.L1Loss', generated_inputs['torch.nn.L1Loss'], lib="torch", suffix=0)
