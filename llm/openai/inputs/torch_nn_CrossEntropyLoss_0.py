
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def cross_entropy_loss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 5).astype(np.float32)
    target1 = np.random.randint(0, 5, size=(3,)).astype(np.int64)
    input_dict1 = {
        "weight": np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32),
        "size_average": True,
        "ignore_index": -100,
        "reduce": True,
        "reduction": "mean",
        "label_smoothing": 0.1,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(4, 3).astype(np.float32)
    target2 = np.random.randint(0, 3, size=(4,)).astype(np.int64)
    input_dict2 = {
        "weight": np.array([0.5, 1.5, 2.5]).astype(np.float32),
        "size_average": False,
        "ignore_index": 2,
        "reduce": True,
        "reduction": "sum",
        "label_smoothing": 0.0,
        "input": input2,
        "target": target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    return list_of_inputs

generated_inputs["torch.nn.CrossEntropyLoss"] = cross_entropy_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CrossEntropyLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CrossEntropyLoss'.")


check_valid('torch.nn.CrossEntropyLoss', generated_inputs['torch.nn.CrossEntropyLoss'], lib="torch", suffix=0)
