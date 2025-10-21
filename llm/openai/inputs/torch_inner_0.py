
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def inner_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other1 = torch.tensor([0, 2, 1], dtype=torch.int32).numpy()
    out1 = torch.empty(0, dtype=torch.int32).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, 2, 3], dtype=torch.int32).numpy()
    other2 = torch.tensor([0, 2, 1], dtype=torch.int32).numpy()
    out2 = torch.empty(0, dtype=torch.int32).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor(2, dtype=torch.int32).numpy()
    other3 = torch.tensor([0, 2, 1], dtype=torch.int32).numpy()
    out3 = torch.empty(0, dtype=torch.int32).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.inner"] = inner_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.inner' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.inner'.")


check_valid('torch.inner', generated_inputs['torch.inner'], lib="torch", suffix=0)
