
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def binary_cross_entropy_inputs():
    list_of_inputs = []
    
    input1 = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    target1 = np.array([0, 1, 0], dtype=np.float32)
    weight1 = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    size_average1 = True
    reduce1 = True
    reduction1 = 'mean'
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "size_average": size_average1,
        "reduce": reduce1,
        "reduction": reduction1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
    target2 = np.array([[0, 1], [1, 0]], dtype=np.float32)
    weight2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    size_average2 = False
    reduce2 = False
    reduction2 = 'sum'
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "size_average": size_average2,
        "reduce": reduce2,
        "reduction": reduction2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([0.7, 0.3, 0.9, 0.1], dtype=np.float32)
    target3 = np.array([1, 0, 1, 0], dtype=np.float32)
    weight3 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    size_average3 = True
    reduce3 = True
    reduction3 = 'none'
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "size_average": size_average3,
        "reduce": reduce3,
        "reduction": reduction3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.nn.functional.binary_cross_entropy"] = binary_cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.binary_cross_entropy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.binary_cross_entropy'.")


check_valid('torch.nn.functional.binary_cross_entropy', generated_inputs['torch.nn.functional.binary_cross_entropy'], lib="torch", suffix=0)
