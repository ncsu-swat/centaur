
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def constant_pad1d_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(1, 2, 4).numpy()
    padding1 = 2
    value1 = 3.5
    input_dict1 = {"padding": padding1, "value": value1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(1, 2, 3).numpy()
    padding2 = (3, 1)
    value2 = 3.5
    input_dict2 = {"padding": padding2, "value": value2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.randn(3, 5).numpy()
    padding3 = 1
    value3 = 0.0
    input_dict3 = {"padding": padding3, "value": value3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2).numpy()
    padding4 = (0, 2)
    value4 = -1.0
    input_dict4 = {"padding": padding4, "value": value4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(1, 1, 5).numpy()
    padding5 = 5
    value5 = 1.0
    input_dict5 = {"padding": padding5, "value": value5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(4, 1).numpy()
    padding6 = (1, 0)
    value6 = 2.5
    input_dict6 = {"padding": padding6, "value": value6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 3, 3).numpy()
    padding7 = 0
    value7 = -0.5
    input_dict7 = {"padding": padding7, "value": value7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 1, 4).numpy()
    padding8 = (2, 2)
    value8 = 4.0
    input_dict8 = {"padding": padding8, "value": value8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.randn(1, 4, 2).numpy()
    padding9 = 3
    value9 = -2.0
    input_dict9 = {"padding": padding9, "value": value9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.randn(3, 2, 1).numpy()
    padding10 = (1, 2)
    value10 = 1.5
    input_dict10 = {"padding": padding10, "value": value10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.ConstantPad1d_1"] = constant_pad1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConstantPad1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConstantPad1d_1'.")


check_valid('torch.nn.ConstantPad1d', generated_inputs['torch.nn.ConstantPad1d_1'], lib="torch", suffix=1)
