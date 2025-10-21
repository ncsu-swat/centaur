
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def copysign_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    other1 = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    out1 = torch.empty_like(input1)
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, dtype=torch.float64)
    other2 = torch.tensor([1, -1, 1, 0, -1, 0]).reshape(2, 3)
    other2 = other2.type(torch.float64)
    out2 = torch.empty_like(input2)
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([-0.0, 0.0, -0.0], dtype=torch.float64)
    other3 = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.float64)
    out3 = torch.empty_like(input3)
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.tensor([1.0], dtype=torch.float64)
    other4 = torch.tensor([-0.0], dtype=torch.float64)
    out4 = torch.empty_like(input4)
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.copysign"] = copysign_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.copysign' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.copysign'.")


check_valid('torch.copysign', generated_inputs['torch.copysign'], lib="torch", suffix=0)
