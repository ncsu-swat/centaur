
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def nllloss_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(3, 5, dtype=torch.float32).numpy()
    target1 = torch.tensor([1, 0, 4], dtype=torch.long).numpy()
    input_dict1 = {
        'weight': np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        'size_average': True,
        'ignore_index': -100,
        'reduce': True,
        'reduction': 'mean',
        'input': input1,
        'target': target1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(2, 3, 4, 4, dtype=torch.float32).numpy()
    target2 = torch.randint(0, 3, (2, 4, 4), dtype=torch.long).numpy()
    input_dict2 = {
        'weight': np.array([0.5, 1.0, 1.5], dtype=np.float32),
        'size_average': False,
        'ignore_index': 0,
        'reduce': False,
        'reduction': 'sum',
        'input': input2,
        'target': target2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(1, 10, dtype=torch.float32).numpy()
    target3 = torch.tensor([5], dtype=torch.long).numpy()
    input_dict3 = {
        'weight': np.array([2.0] * 10, dtype=np.float32),
        'size_average': True,
        'ignore_index': -1,
        'reduce': True,
        'reduction': 'mean',
        'input': input3,
        'target': target3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 2, dtype=torch.float32).numpy()
    target4 = torch.tensor([0, 1, 0, 1], dtype=torch.long).numpy()
    input_dict4 = {
        'weight': np.array([1.0, 2.0], dtype=np.float32),
        'size_average': False,
        'ignore_index': 1,
        'reduce': False,
        'reduction': 'none',
        'input': input4,
        'target': target4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.nn.NLLLoss"] = nllloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.NLLLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.NLLLoss'.")


check_valid('torch.nn.NLLLoss', generated_inputs['torch.nn.NLLLoss'], lib="torch", suffix=0)
