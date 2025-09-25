
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch

def isposinf_inputs():
    list_of_inputs = []
    input1 = torch.tensor([float('inf'), -float('inf'), 1.2])
    out1 = torch.tensor([False, False, False])
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(input_dict1)

    input2 = torch.tensor([[-float('inf'), float('inf')], [1.2, -float('inf')]])
    out2 = torch.tensor([False, False, False, False])
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(input_dict2)

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.isposinf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.isposinf'.")


check_valid('torch.isposinf', generated_inputs['torch.isposinf'], lib="torch", suffix=0)
