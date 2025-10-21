
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def gelu_inputs():
    list_of_inputs = []

    input1 = np.random.rand(2).astype(np.float32)
    input_dict1 = {"approximate": "none", "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 4).astype(np.float32)
    input_dict2 = {"approximate": "tanh", "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(5, 2, 3).astype(np.float32)
    input_dict3 = {"approximate": "none", "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([-1.0, 0.0, 1.0]).astype(np.float32)
    input_dict4 = {"approximate": "tanh", "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(1).astype(np.float32)
    input_dict5 = {"approximate": "none", "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(4, 4).astype(np.float32)
    input_dict6 = {"approximate": "tanh", "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict7 = {"approximate": "none", "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[-2.0, 3.0], [1.0, -1.5]]).astype(np.float32)
    input_dict8 = {"approximate": "tanh", "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(10, 1).astype(np.float32)
    input_dict9 = {"approximate": "none", "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 3, 1).astype(np.float32)
    input_dict10 = {"approximate": "tanh", "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.GELU"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GELU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GELU'.")


check_valid('torch.nn.GELU', generated_inputs['torch.nn.GELU'], lib="torch", suffix=0)
