
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gaussian_nll_loss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(5, 2).astype(np.float32)
    target1 = np.random.rand(5, 2).astype(np.float32)
    var1 = np.ones((5, 2)).astype(np.float32)
    input_dict1 = {
        "full": False,
        "eps": 1e-6,
        "reduction": "mean",
        "input": input1,
        "target": target1,
        "var": var1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 4, 5).astype(np.float32)
    target2 = np.random.rand(3, 4, 5).astype(np.float32)
    var2 = np.full((3, 4, 5), 0.5).astype(np.float32)
    input_dict2 = {
        "full": True,
        "eps": 1e-7,
        "reduction": "sum",
        "input": input2,
        "target": target2,
        "var": var2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 2).astype(np.float32)
    target3 = np.random.rand(2, 2).astype(np.float32)
    var3 = np.ones((2,2)).astype(np.float32)
    input_dict3 = {
        "full": False,
        "eps": 1e-5,
        "reduction": "none",
        "input": input3,
        "target": target3,
        "var": var3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(4, 3).astype(np.float32)
    target4 = np.random.rand(4, 3).astype(np.float32)
    var4 = np.random.rand(4, 3).astype(np.float32)
    input_dict4 = {
        "full": True,
        "eps": 1e-8,
        "reduction": "mean",
        "input": input4,
        "target": target4,
        "var": var4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(10,).astype(np.float32)
    target5 = np.random.rand(10,).astype(np.float32)
    var5 = np.ones(10).astype(np.float32)
    input_dict5 = {
        "full": False,
        "eps": 1e-6,
        "reduction": "sum",
        "input": input5,
        "target": target5,
        "var": var5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 3).astype(np.float32)
    target6 = np.random.rand(2, 3).astype(np.float32)
    var6 = np.ones((2,3)).astype(np.float32)
    input_dict6 = {
        "full": False,
        "eps": 1e-6,
        "reduction": "mean",
        "input": input6,
        "target": target6,
        "var": var6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.GaussianNLLLoss"] = gaussian_nll_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GaussianNLLLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GaussianNLLLoss'.")


check_valid('torch.nn.GaussianNLLLoss', generated_inputs['torch.nn.GaussianNLLLoss'], lib="torch", suffix=0)
