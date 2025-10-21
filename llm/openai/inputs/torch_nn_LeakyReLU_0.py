
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def leaky_relu_inputs():
    list_of_inputs = []

    input1 = np.random.rand(2)
    input_dict1 = {
        "negative_slope": 0.1,
        "inplace": True,
        "input": torch.tensor(input1).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(3, 4)
    input_dict2 = {
        "negative_slope": 0.01,
        "inplace": False,
        "input": torch.tensor(input2).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(5, 2, 3)
    input_dict3 = {
        "negative_slope": 0.2,
        "inplace": True,
        "input": torch.tensor(input3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1)
    input_dict4 = {
        "negative_slope": -0.1,
        "inplace": False,
        "input": torch.tensor(input4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(4, 4)
    input_dict5 = {
        "negative_slope": 0.0,
        "inplace": True,
        "input": torch.tensor(input5).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 2, 2, 2)
    input_dict6 = {
        "negative_slope": 0.3,
        "inplace": False,
        "input": torch.tensor(input6).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(6)
    input_dict7 = {
        "negative_slope": -0.05,
        "inplace": True,
        "input": torch.tensor(input7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(1, 5)
    input_dict8 = {
        "negative_slope": 0.15,
        "inplace": False,
        "input": torch.tensor(input8).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(7, 3)
    input_dict9 = {
        "negative_slope": 0.02,
        "inplace": True,
        "input": torch.tensor(input9).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(2, 3, 1)
    input_dict10 = {
        "negative_slope": -0.2,
        "inplace": False,
        "input": torch.tensor(input10).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.LeakyReLU"] = leaky_relu_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LeakyReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LeakyReLU'.")


check_valid('torch.nn.LeakyReLU', generated_inputs['torch.nn.LeakyReLU'], lib="torch", suffix=0)
