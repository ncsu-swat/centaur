
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def sigmoid_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(2).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = torch.tensor([[-1.0, 0.0, 1.0]]).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(2, 2, 2).numpy()
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.tensor([[-2.5, -1.0, 0.0, 1.0, 2.5]]).numpy()
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(5, 1).numpy()
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.tensor([0.0]).numpy()
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(1, 3, 3, 3).numpy()
    input_dict8 = {"input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([-5.0, 5.0, -10.0, 10.0]).numpy()
    input_dict9 = {"input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = torch.randn(4, 4, dtype=torch.float32).numpy()
    input_dict10 = {"input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.Sigmoid"] = sigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Sigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Sigmoid'.")


check_valid('torch.nn.Sigmoid', generated_inputs['torch.nn.Sigmoid'], lib="torch", suffix=0)
