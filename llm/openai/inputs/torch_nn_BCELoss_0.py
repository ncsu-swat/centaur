
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bceloss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 2).astype(np.float32)
    target1 = np.random.rand(3, 2).astype(np.float32)
    input_dict1 = {
        "weight": np.array([1.0, 2.0]).astype(np.float32),
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(4, 4, 2).astype(np.float32)
    target2 = np.random.rand(4, 4, 2).astype(np.float32)
    input_dict2 = {
        "weight": np.array([0.5, 1.5]).astype(np.float32),
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.random.rand(2, 5).astype(np.float32)
    target3 = np.random.rand(2, 5).astype(np.float32)
    input_dict3 = {
        "weight": None,
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 1).astype(np.float32)
    target4 = np.random.rand(1, 1).astype(np.float32)
    input_dict4 = {
        "weight": np.array([1.0]).astype(np.float32),
        "size_average": False,
        "reduce": False,
        "reduction": "mean",
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs["torch.nn.BCELoss"] = bceloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BCELoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BCELoss'.")


check_valid('torch.nn.BCELoss', generated_inputs['torch.nn.BCELoss'], lib="torch", suffix=0)
