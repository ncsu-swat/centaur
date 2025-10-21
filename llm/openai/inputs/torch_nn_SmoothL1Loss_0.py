
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smooth_l1_loss_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict1 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1.0,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    input_dict2 = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "beta": 0.5,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    target3 = np.array([0.0, -1.0, 2.0], dtype=np.float32)
    input_dict3 = {
        "size_average": True,
        "reduce": True,
        "reduction": "none",
        "beta": 2.0,
        "input": input3,
        "target": target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    target4 = np.array([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]], dtype=np.float32)
    input_dict4 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1.5,
        "input": input4,
        "target": target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0], dtype=np.float32)
    target5 = np.array([3.0, 4.0], dtype=np.float32)
    input_dict5 = {
        "size_average": False,
        "reduce": False,
        "reduction": "sum",
        "beta": 0.1,
        "input": input5,
        "target": target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([1.0], dtype=np.float32)
    target6 = np.array([2.0], dtype=np.float32)
    input_dict6 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 0.0,
        "input": input6,
        "target": target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    target7 = np.array([11.0, 22.0, 33.0], dtype=np.float32)
    input_dict7 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 5.0,
        "input": input7,
        "target": target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    target8 = np.array([[-1.5, 2.5], [-3.5, 4.5]], dtype=np.float32)
    input_dict8 = {
        "size_average": False,
        "reduce": False,
        "reduction": "none",
        "beta": 1.0,
        "input": input8,
        "target": target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict9 = {
        "size_average": True,
        "reduce": True,
        "reduction": "mean",
        "beta": 1.0,
        "input": input9,
        "target": target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    target10 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict10 = {
        "size_average": True,
        "reduce": True,
        "reduction": "sum",
        "beta": 1.0,
        "input": input10,
        "target": target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.SmoothL1Loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.SmoothL1Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.SmoothL1Loss'.")


check_valid('torch.nn.SmoothL1Loss', generated_inputs['torch.nn.SmoothL1Loss'], lib="torch", suffix=0)
