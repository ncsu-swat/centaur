
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def det_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 3).numpy()
    out1 = torch.empty(1).numpy()
    input_dict1 = {"A": input1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2).numpy()
    out2 = torch.empty(1).numpy()
    input_dict2 = {"A": input2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 3, 3).numpy()
    out3 = torch.empty(1).numpy()
    input_dict3 = {"A": input3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2).numpy()
    out4 = torch.empty(1).numpy()
    input_dict4 = {"A": input4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 4).numpy()
    out5 = torch.empty(1).numpy()
    input_dict5 = {"A": input5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(5, 5).numpy()
    out6 = torch.empty(1).numpy()
    input_dict6 = {"A": input6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 3, 3).numpy()
    out7 = torch.empty(1).numpy()
    input_dict7 = {"A": input7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 3, 3).numpy()
    out8 = torch.empty(1).numpy()
    input_dict8 = {"A": input8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(3, 2, 2).numpy()
    out9 = torch.empty(1).numpy()
    input_dict9 = {"A": input9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    

    return list_of_inputs

generated_inputs["torch.linalg.det"] = det_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.det' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.det'.")


check_valid('torch.linalg.det', generated_inputs['torch.linalg.det'], lib="torch", suffix=0)
