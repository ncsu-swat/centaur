
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def floor_divide_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()
    other1 = torch.tensor([2, 4, 6], dtype=torch.int32).numpy()
    out1 = torch.tensor([], dtype=torch.int32).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([-10, -20, -30], dtype=torch.int32).numpy()
    other2 = torch.tensor([2, 4, 6], dtype=torch.int32).numpy()
    out2 = torch.tensor([], dtype=torch.int32).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()
    other3 = torch.tensor([2], dtype=torch.int32).numpy()
    out3 = torch.tensor([], dtype=torch.int32).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[10, 20], [30, 40]], dtype=torch.int32).numpy()
    other4 = torch.tensor([[2, 4], [6, 8]], dtype=torch.int32).numpy()
    out4 = torch.tensor([], dtype=torch.int32).numpy()
    input_dict4 = {"input": input4, "other": other4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([10, 20, 30], dtype=torch.int32).numpy()
    other5 = torch.tensor([3], dtype=torch.int32).numpy()
    out5 = torch.tensor([], dtype=torch.int32).numpy()
    input_dict5 = {"input": input5, "other": other5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.floor_divide_1'.")


check_valid('torch.floor_divide', generated_inputs['torch.floor_divide_1'], lib="torch", suffix=1)
