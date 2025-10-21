
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []

    input1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict1 = {"input": input1.numpy(), "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict2 = {"input": input2.numpy(), "start_dim": 1, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input_dict3 = {"input": input3.numpy(), "start_dim": 0, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([1, 2, 3, 4, 5, 6])
    input_dict4 = {"input": input4.numpy(), "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([[1, 2], [3, 4]])
    input_dict5 = {"input": input5.numpy(), "start_dim": 1, "end_dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1])
    input_dict6 = {"input": input6.numpy(), "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 3, 4, 5)
    input_dict7 = {"input": input7.numpy(), "start_dim": 1, "end_dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 3, 4, 5)
    input_dict8 = {"input": input8.numpy(), "start_dim": -3, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(1, 2, 3)
    input_dict9 = {"input": input9.numpy(), "start_dim": 0, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(4, 1, 1, 1)
    input_dict10 = {"input": input10.numpy(), "start_dim": 0, "end_dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.flatten' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.flatten'.")


check_valid('torch.flatten', generated_inputs['torch.flatten'], lib="torch", suffix=0)
