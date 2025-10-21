
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def smooth_l1_loss_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    target1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    reduction1 = "mean"
    delta1 = 1.0
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "reduction": reduction1,
        "delta": delta1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.functional.smooth_l1_loss"] = smooth_l1_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.smooth_l1_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.smooth_l1_loss'.")


check_valid('torch.nn.functional.smooth_l1_loss', generated_inputs['torch.nn.functional.smooth_l1_loss'], lib="torch", suffix=0)
