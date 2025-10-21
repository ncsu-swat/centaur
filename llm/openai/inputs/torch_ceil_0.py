
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ceil_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1.2, 2.7, 3.1]).numpy()
    out1 = torch.empty(0, dtype=torch.float).numpy()
    input_dict1 = {"input": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([-1.2, -2.7, -3.1]).numpy()
    out2 = torch.empty(0, dtype=torch.float).numpy()
    input_dict2 = {"input": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([0.0, -0.5, 0.9]).numpy()
    out3 = torch.empty(0, dtype=torch.float).numpy()
    input_dict3 = {"input": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([[1.1, 2.2], [3.3, 4.4]]).numpy()
    out4 = torch.empty(0, dtype=torch.float).numpy()
    input_dict4 = {"input": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([[-1.1, -2.2], [-3.3, -4.4]]).numpy()
    out5 = torch.empty(0, dtype=torch.float).numpy()
    input_dict5 = {"input": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.ceil'.")


check_valid('torch.ceil', generated_inputs['torch.ceil'], lib="torch", suffix=0)
