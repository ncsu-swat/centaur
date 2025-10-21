
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def dot_inputs():
    list_of_inputs = []
    
    input1 = torch.tensor([1, 2, 3]).numpy()
    tensor1 = torch.tensor([4, 5, 6]).numpy()
    out1 = torch.tensor(0).numpy()
    input_dict1 = {"input": input1, "tensor": tensor1, "out": out1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.tensor([-1, -2, -3]).numpy()
    tensor2 = torch.tensor([1, 2, 3]).numpy()
    out2 = torch.tensor(0).numpy()
    input_dict2 = {"input": input2, "tensor": tensor2, "out": out2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([0, 0, 0]).numpy()
    tensor3 = torch.tensor([1, 2, 3]).numpy()
    out3 = torch.tensor(0).numpy()
    input_dict3 = {"input": input3, "tensor": tensor3, "out": out3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor4 = torch.tensor([4.0, 5.0, 6.0]).numpy()
    out4 = torch.tensor(0.0).numpy()
    input_dict4 = {"input": input4, "tensor": tensor4, "out": out4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([1, 2, 3]).numpy()
    tensor5 = torch.tensor([6, 5, 4]).numpy()
    out5 = torch.tensor(0).numpy()
    input_dict5 = {"input": input5, "tensor": tensor5, "out": out5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.tensor([1, -2, 3]).numpy()
    tensor6 = torch.tensor([-4, 5, -6]).numpy()
    out6 = torch.tensor(0).numpy()
    input_dict6 = {"input": input6, "tensor": tensor6, "out": out6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.tensor([10, 20, 30]).numpy()
    tensor7 = torch.tensor([1, 2, 3]).numpy()
    out7 = torch.tensor(0).numpy()
    input_dict7 = {"input": input7, "tensor": tensor7, "out": out7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([1, 1, 1]).numpy()
    tensor8 = torch.tensor([1, 1, 1]).numpy()
    out8 = torch.tensor(0).numpy()
    input_dict8 = {"input": input8, "tensor": tensor8, "out": out8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.tensor([5, -5, 5]).numpy()
    tensor9 = torch.tensor([-5, 5, -5]).numpy()
    out9 = torch.tensor(0).numpy()
    input_dict9 = {"input": input9, "tensor": tensor9, "out": out9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = torch.tensor([2, 4, 6]).numpy()
    tensor10 = torch.tensor([1, 3, 5]).numpy()
    out10 = torch.tensor(0).numpy()
    input_dict10 = {"input": input10, "tensor": tensor10, "out": out10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.dot"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.dot' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dot'.")


check_valid('torch.dot', generated_inputs['torch.dot'], lib="torch", suffix=0)
