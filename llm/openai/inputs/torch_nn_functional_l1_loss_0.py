
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def l1_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    target1 = np.array([1.0, 2.0, 3.0])
    reduction1 = "mean"
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    target2 = np.array([[1.0, 2.0], [3.0, 5.0]])
    reduction2 = "sum"
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-1.0, 2.0], [3.0, -4.0]])
    target3 = np.array([[1.0, -2.0], [-3.0, 4.0]])
    reduction3 = "mean"
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0])
    target4 = np.array([2.0])
    reduction4 = "sum"

    input_dict4 = {
        "input": input4,
        "target": target4,
        "reduction": reduction4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0, 2.0, 3.0, 4.0])
    target5 = np.array([5.0, 6.0, 7.0, 8.0])
    reduction5 = "mean"

    input_dict5 = {
        "input": input5,
        "target": target5,
        "reduction": reduction5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    target6 = np.array([[[1.0, 2.0], [3.0, 5.0]], [[5.0, 7.0], [7.0, 9.0]]])
    reduction6 = "sum"

    input_dict6 = {
        "input": input6,
        "target": target6,
        "reduction": reduction6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1.0, -2.0], [-3.0, 4.0]])
    target7 = np.array([[1.0, 2.0], [3.0, -4.0]])
    reduction7 = "mean"

    input_dict7 = {
        "input": input7,
        "target": target7,
        "reduction": reduction7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([0.0, 0.0, 0.0])
    target8 = np.array([0.0, 0.0, 0.0])
    reduction8 = "sum"

    input_dict8 = {
        "input": input8,
        "target": target8,
        "reduction": reduction8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.5, 2.5, 3.5])
    target9 = np.array([1.0, 2.0, 3.0])
    reduction9 = "mean"

    input_dict9 = {
        "input": input9,
        "target": target9,
        "reduction": reduction9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([[1.0, 2.0, 3.0]])
    target10 = np.array([[4.0, 5.0, 6.0]])
    reduction10 = "sum"
    
    input_dict10 = {
        "input": input10,
        "target": target10,
        "reduction": reduction10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.l1_loss"] = l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.l1_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.l1_loss'.")


check_valid('torch.nn.functional.l1_loss', generated_inputs['torch.nn.functional.l1_loss'], lib="torch", suffix=0)
