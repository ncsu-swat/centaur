
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def kl_div_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.1, 0.2, 0.7])
    target1 = np.array([0.2, 0.3, 0.5])
    reduction1 = 'sum'
    log_target1 = False
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "reduction": reduction1,
        "log_target": log_target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[0.1, 0.2], [0.3, 0.4]])
    target2 = np.array([[0.2, 0.3], [0.4, 0.5]])
    reduction2 = 'mean'
    log_target2 = True
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "reduction": reduction2,
        "log_target": log_target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[-0.1, 0.2], [0.3, -0.4]])
    target3 = np.array([[0.2, 0.3], [0.4, 0.5]])
    reduction3 = 'sum'
    log_target3 = False
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "reduction": reduction3,
        "log_target": log_target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([0.1, 0.2, 0.3, 0.4])
    target4 = np.array([0.2, 0.3, 0.4, 0.1])
    reduction4 = 'batchmean'
    log_target4 = True
    
    input_dict4 = {
        "input": input4,
        "target": target4,
        "reduction": reduction4,
        "log_target": log_target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([0.01, 0.02, 0.03])
    target5 = np.array([0.1, 0.2, 0.7])
    reduction5 = 'mean'
    log_target5 = False
    
    input_dict5 = {
        "input": input5,
        "target": target5,
        "reduction": reduction5,
        "log_target": log_target5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([[0.1, 0.9], [0.8, 0.2]])
    target6 = np.array([[0.5, 0.5], [0.5, 0.5]])
    reduction6 = 'sum'
    log_target6 = True

    input_dict6 = {
        "input": input6,
        "target": target6,
        "reduction": reduction6,
        "log_target": log_target6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([1.0, 2.0, 3.0])
    target7 = np.array([0.5, 1.5, 2.5])
    reduction7 = 'mean'
    log_target7 = False

    input_dict7 = {
        "input": input7,
        "target": target7,
        "reduction": reduction7,
        "log_target": log_target7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    target8 = np.array([[0.2, 0.3, 0.1], [0.5, 0.4, 0.1]])
    reduction8 = 'batchmean'
    log_target8 = True

    input_dict8 = {
        "input": input8,
        "target": target8,
        "reduction": reduction8,
        "log_target": log_target8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([0.001, 0.002, 0.003])
    target9 = np.array([0.1, 0.2, 0.3])
    reduction9 = 'sum'
    log_target9 = False

    input_dict9 = {
        "input": input9,
        "target": target9,
        "reduction": reduction9,
        "log_target": log_target9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([0.5, 0.5, 0.0])
    target10 = np.array([0.2, 0.3, 0.5])
    reduction10 = 'mean'
    log_target10 = True
    
    input_dict10 = {
        "input": input10,
        "target": target10,
        "reduction": reduction10,
        "log_target": log_target10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.kl_div"] = kl_div_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.kl_div' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.kl_div'.")


check_valid('torch.nn.functional.kl_div', generated_inputs['torch.nn.functional.kl_div'], lib="torch", suffix=0)
