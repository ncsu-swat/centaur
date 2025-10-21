
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_same_size_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    target1 = torch.tensor([4, 5, 6]).numpy()
    input_dict = {"input": input1, "target": target1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input2 = torch.zeros((2, 3)).numpy()
    target2 = torch.ones((2, 3)).numpy()
    input_dict = {"input": input2, "target": target2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input3 = torch.randn(4, 4, 4).numpy()
    target3 = torch.randint(0, 10, (4, 4, 4)).numpy()
    input_dict = {"input": input3, "target": target3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    target4 = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).numpy()
    input_dict = {"input": input4, "target": target4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input5 = torch.tensor([[-1, -2, -3]]).numpy()
    target5 = torch.tensor([[1, 2, 3]]).numpy()
    input_dict = {"input": input5, "target": target5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input6 = torch.tensor([]).numpy()
    target6 = torch.tensor([]).numpy()
    input_dict = {"input": input6, "target": target6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input7 = torch.tensor([1]).numpy()
    target7 = torch.tensor([1]).numpy()
    input_dict = {"input": input7, "target": target7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input8 = torch.tensor([[1, 2], [3, 4]]).numpy()
    target8 = torch.tensor([[5, 6], [7, 8]]).numpy()
    input_dict = {"input": input8, "target": target8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    input9 = torch.randn(2, 2, 2, 2).numpy()
    target9 = torch.randint(0, 10, (2, 2, 2, 2)).numpy()
    input_dict = {"input": input9, "target": target9}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input10 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    target10 = torch.tensor([6, 7, 8, 9, 10]).numpy()
    input_dict = {"input": input10, "target": target10}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.is_same_size"] = is_same_size_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_same_size' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_same_size'.")


check_valid('torch.is_same_size', generated_inputs['torch.is_same_size'], lib="torch", suffix=0)
