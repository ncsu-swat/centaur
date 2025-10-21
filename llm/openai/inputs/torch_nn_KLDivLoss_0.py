
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kl_div_loss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(2, 3).astype(np.float32)
    target1 = np.random.rand(2, 3).astype(np.float32)
    input_dict1 = {
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "log_target": False,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, 4).astype(np.float32)
    target2 = np.random.rand(4, 4).astype(np.float32)
    input_dict2 = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "log_target": False,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(3, 5).astype(np.float32)
    target3 = np.random.rand(3, 5).astype(np.float32)
    input_dict3 = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "log_target": True,
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 1).astype(np.float32)
    target4 = np.random.rand(1, 1).astype(np.float32)
    input_dict4 = {
        "size_average": False,
        "reduce": True,
        "reduction": "batchmean",
        "log_target": False,
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(5, 2).astype(np.float32)
    target5 = np.random.rand(5, 2).astype(np.float32)
    input_dict5 = {
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "log_target": True,
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 2, 2).astype(np.float32)
    target6 = np.random.rand(2, 2, 2).astype(np.float32)
    input_dict6 = {
        "size_average": False,
        "reduce": True,
        "reduction": "sum",
        "log_target": False,
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(10,).astype(np.float32)
    target7 = np.random.rand(10,).astype(np.float32)
    input_dict7 = {
        "size_average": True,
        "reduce": False,
        "reduction": "mean",
        "log_target": False,
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 3).astype(np.float32)
    target8 = np.random.rand(3, 3).astype(np.float32)
    input_dict8 = {
        "size_average": False,
        "reduce": True,
        "reduction": "batchmean",
        "log_target": True,
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(2, 5).astype(np.float32)
    target9 = np.random.rand(2, 5).astype(np.float32)
    input_dict9 = {
        "size_average": True,
        "reduce": False,
        "reduction": "none",
        "log_target": False,
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.random.rand(6, 1).astype(np.float32)
    target10 = np.random.rand(6, 1).astype(np.float32)
    input_dict10 = {
        "size_average": False,
        "reduce": True,
        "reduction": "mean",
        "log_target": True,
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.KLDivLoss"] = kl_div_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.KLDivLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.KLDivLoss'.")


check_valid('torch.nn.KLDivLoss', generated_inputs['torch.nn.KLDivLoss'], lib="torch", suffix=0)
