
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def hinge_embedding_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.5, -0.2, 0.1, -0.8]).astype(np.float32)
    target1 = np.array([1, -1, 1, -1]).astype(np.float32)
    margin1 = 0.7
    reduction1 = 'mean'
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "margin": margin1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[0.2, -0.9], [0.6, -0.1]]).astype(np.float32)
    target2 = np.array([[1, -1], [-1, 1]]).astype(np.float32)
    margin2 = 1.0
    reduction2 = 'sum'
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "margin": margin2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[-0.3], [0.4], [-0.1]]).astype(np.float32)
    target3 = np.array([-1, 1, -1]).astype(np.float32)
    margin3 = 0.2
    reduction3 = 'none'
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "margin": margin3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.9, -0.5, 0.7, -0.3, 0.1]).astype(np.float32)
    target4 = np.array([1, -1, 1, -1, 1]).astype(np.float32)
    margin4 = 0.5
    reduction4 = 'mean'
    
    input_dict4 = {
        "input": input4,
        "target": target4,
        "margin": margin4,
        "reduction": reduction4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[0.1, 0.2], [-0.3, -0.4]]).astype(np.float32)
    target5 = np.array([[1, -1], [-1, 1]]).astype(np.float32)
    margin5 = 0.8
    reduction5 = 'sum'
    
    input_dict5 = {
        "input": input5,
        "target": target5,
        "margin": margin5,
        "reduction": reduction5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    
    return list_of_inputs

generated_inputs["torch.nn.functional.hinge_embedding_loss"] = hinge_embedding_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.hinge_embedding_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hinge_embedding_loss'.")


check_valid('torch.nn.functional.hinge_embedding_loss', generated_inputs['torch.nn.functional.hinge_embedding_loss'], lib="torch", suffix=0)
