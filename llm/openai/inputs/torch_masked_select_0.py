
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def masked_select_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    mask1 = torch.tensor([True, False, True, False, True]).numpy()
    input_dict1 = {"input": input1, "mask": mask1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    mask2 = torch.tensor([[True, False], [False, True]]).numpy()
    input_dict2 = {"input": input2, "mask": mask2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(3, 4).numpy()
    mask3 = torch.randint(0, 2, (3, 4)).numpy().astype(bool)
    input_dict3 = {"input": input3, "mask": mask3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.tensor([-1, -2, -3, -4, -5]).numpy()
    mask4 = torch.tensor([False, True, False, True, False]).numpy()
    input_dict4 = {"input": input4, "mask": mask4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    mask5 = torch.tensor([True, True, False, False, True]).numpy()
    input_dict5 = {"input": input5, "mask": mask5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1, 0, 1, 0, 1]).numpy()
    mask6 = torch.tensor([0, 1, 0, 1, 0]).numpy().astype(bool)
    input_dict6 = {"input": input6, "mask": mask6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 2, 2).numpy()
    mask7 = torch.randint(0, 2, (2, 2, 2)).numpy().astype(bool)
    input_dict7 = {"input": input7, "mask": mask7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = torch.tensor([10, 20, 30, 40]).numpy()
    mask8 = torch.tensor([True, False, True, True]).numpy()
    input_dict8 = {"input": input8, "mask": mask8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([1, 2, 3]).numpy()
    mask9 = torch.tensor([False, False, False]).numpy()
    input_dict9 = {"input": input9, "mask": mask9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.tensor([5, 10, 15, 20]).numpy()
    mask10 = torch.tensor([True, True, False, False]).numpy()
    input_dict10 = {"input": input10, "mask": mask10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.masked_select"] = masked_select_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.masked_select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.masked_select'.")


check_valid('torch.masked_select', generated_inputs['torch.masked_select'], lib="torch", suffix=0)
