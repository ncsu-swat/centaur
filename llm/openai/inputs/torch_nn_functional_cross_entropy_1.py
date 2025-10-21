
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_inputs():
    list_of_inputs = []
    
    input1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    target1 = np.array([0, 1], dtype=np.int64)
    weight1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ignore_index1 = -1
    reduction1 = 'mean'
    label_smoothing1 = 0.1
    
    input_dict1 = {
        "input": input1,
        "target": target1,
        "weight": weight1,
        "ignore_index": ignore_index1,
        "reduction": reduction1,
        "label_smoothing": label_smoothing1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    target2 = np.array([1, 2], dtype=np.int64)
    weight2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ignore_index2 = 0
    reduction2 = 'sum'
    label_smoothing2 = 0.0
    
    input_dict2 = {
        "input": input2,
        "target": target2,
        "weight": weight2,
        "ignore_index": ignore_index2,
        "reduction": reduction2,
        "label_smoothing": label_smoothing2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    target3 = np.array([0, 1, 0], dtype=np.int64)
    weight3 = np.array([1.0, 1.0], dtype=np.float32)
    ignore_index3 = -1
    reduction3 = 'mean'
    label_smoothing3 = 0.0
    
    input_dict3 = {
        "input": input3,
        "target": target3,
        "weight": weight3,
        "ignore_index": ignore_index3,
        "reduction": reduction3,
        "label_smoothing": label_smoothing3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.5]], dtype=np.float32)
    target4 = np.array([0, 0], dtype=np.int64)
    weight4 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ignore_index4 = -1
    reduction4 = 'mean'
    label_smoothing4 = 0.2
    
    input_dict4 = {
        "input": input4,
        "target": target4,
        "weight": weight4,
        "ignore_index": ignore_index4,
        "reduction": reduction4,
        "label_smoothing": label_smoothing4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.functional.cross_entropy_1"] = cross_entropy_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.cross_entropy_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.cross_entropy_1'.")


check_valid('torch.nn.functional.cross_entropy', generated_inputs['torch.nn.functional.cross_entropy_1'], lib="torch", suffix=1)
