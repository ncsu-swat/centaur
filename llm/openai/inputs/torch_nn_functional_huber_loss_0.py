
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def huber_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    target1 = np.array([1.5, 2.5, 3.5])
    delta1 = 1.0
    reduction1 = "mean"
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "delta": delta1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    target2 = np.array([[1.5, 2.5], [3.5, 4.5]])
    delta2 = 0.5
    reduction2 = "sum"
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "delta": delta2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([-1.0, -2.0, -3.0])
    target3 = np.array([-1.5, -2.5, -3.5])
    delta3 = 1.0
    reduction3 = "mean"
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "delta": delta3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0, 0.0, 0.0])
    target4 = np.array([0.0, 0.0, 0.0])
    delta4 = 2.0
    reduction4 = "none"
    
    input_dict4 = {
        "input": input4,
        "target": target4,
        "delta": delta4,
        "reduction": reduction4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1.0])
    target5 = np.array([1.5])
    delta5 = 0.1
    reduction5 = "mean"
    
    input_dict5 = {
        "input": input5,
        "target": target5,
        "delta": delta5,
        "reduction": reduction5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target6 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    delta6 = 1.5
    reduction6 = "mean"

    input_dict6 = {
        "input": input6,
        "target": target6,
        "delta": delta6,
        "reduction": reduction6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target7 = np.array([[1.2, 2.1, 3.2], [4.3, 5.4, 6.5]])
    delta7 = 0.2
    reduction7 = "sum"

    input_dict7 = {
        "input": input7,
        "target": target7,
        "delta": delta7,
        "reduction": reduction7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, -2.0, 3.0])
    target8 = np.array([1.1, -2.2, 3.3])
    delta8 = 0.5
    reduction8 = "mean"

    input_dict8 = {
        "input": input8,
        "target": target8,
        "delta": delta8,
        "reduction": reduction8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.array([1.0, 2.0])
    target9 = np.array([3.0, 4.0])
    delta9 = 2.0
    reduction9 = "none"

    input_dict9 = {
        "input": input9,
        "target": target9,
        "delta": delta9,
        "reduction": reduction9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target10 = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
    delta10 = 1.0
    reduction10 = "mean"

    input_dict10 = {
        "input": input10,
        "target": target10,
        "delta": delta10,
        "reduction": reduction10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.functional.huber_loss"] = huber_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.huber_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.huber_loss'.")


check_valid('torch.nn.functional.huber_loss', generated_inputs['torch.nn.functional.huber_loss'], lib="torch", suffix=0)
