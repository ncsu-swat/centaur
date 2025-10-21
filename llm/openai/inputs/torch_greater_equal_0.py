
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def greater_equal_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    other1 = torch.tensor([1, 2, 2]).numpy()
    out1 = torch.empty(input1.shape).numpy()
    input_dict1 = {"input": input1, "other": other1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([4, 5, 6]).numpy()
    other2 = torch.tensor([7, 8, 9]).numpy()
    out2 = torch.empty(input2.shape).numpy()
    input_dict2 = {"input": input2, "other": other2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[1, 2], [3, 4]]).numpy()
    other3 = torch.tensor([[1, 2], [3, 3]]).numpy()
    out3 = torch.empty(input3.shape).numpy()
    input_dict3 = {"input": input3, "other": other3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.greater_equal"] = greater_equal_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.greater_equal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.greater_equal'.")


check_valid('torch.greater_equal', generated_inputs['torch.greater_equal'], lib="torch", suffix=0)
