
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_tensor_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    list_of_inputs.append({"obj": input1})
    
    input2 = torch.tensor([[1, 2], [3, 4]]).numpy()
    list_of_inputs.append({"obj": input2})
    
    input3 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    list_of_inputs.append({"obj": input3})
    
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    list_of_inputs.append({"obj": input4})
    
    input5 = torch.tensor([-1, -2, -3]).numpy()
    list_of_inputs.append({"obj": input5})
    
    input6 = torch.tensor([0]).numpy()
    list_of_inputs.append({"obj": input6})
    
    input7 = torch.tensor([1, 2, 3, 4, 5]).numpy()
    list_of_inputs.append({"obj": input7})
    
    input8 = torch.randn(2, 3, 4).numpy()
    list_of_inputs.append({"obj": input8})
    
    input9 = torch.zeros((3, 2)).numpy()
    list_of_inputs.append({"obj": input9})

    input10 = torch.ones((1, 1, 1)).numpy()
    list_of_inputs.append({"obj": input10})
    
    return list_of_inputs

generated_inputs["torch.is_tensor"] = is_tensor_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_tensor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_tensor'.")


check_valid('torch.is_tensor', generated_inputs['torch.is_tensor'], lib="torch", suffix=0)
