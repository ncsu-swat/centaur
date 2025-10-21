
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def rot90_inputs():
    list_of_inputs = []

    input1 = np.arange(4).reshape(2, 2)
    k1 = 1
    dims1 = (0, 1)
    input_dict1 = {"input": input1, "k": k1, "dims": dims1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.arange(8).reshape(2, 2, 2)
    k2 = 1
    dims2 = (1, 2)
    input_dict2 = {"input": input2, "k": k2, "dims": dims2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.arange(12).reshape(3, 2, 2)
    k3 = 2
    dims3 = (0, 1)
    input_dict3 = {"input": input3, "k": k3, "dims": dims3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(4, 4, 4)
    k4 = -1
    dims4 = (1, 2)
    input_dict4 = {"input": input4, "k": k4, "dims": dims4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.zeros((2, 3, 4, 5))
    k5 = 3
    dims5 = (0, 2)
    input_dict5 = {"input": input5, "k": k5, "dims": dims5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.ones((5, 5))
    k6 = -2
    dims6 = (0, 1)
    input_dict6 = {"input": input6, "k": k6, "dims": dims6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.arange(24).reshape(2, 3, 4)
    k7 = 1
    dims7 = (0, 2)
    input_dict7 = {"input": input7, "k": k7, "dims": dims7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(3, 3, 3, 3)
    k8 = 0
    dims8 = (1, 3)
    input_dict8 = {"input": input8, "k": k8, "dims": dims8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.arange(36).reshape(6, 6)
    k9 = -1
    dims9 = (0, 1)
    input_dict9 = {"input": input9, "k": k9, "dims": dims9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.ones((2, 2, 2, 2, 2))
    k10 = 2
    dims10 = (1, 4)
    input_dict10 = {"input": input10, "k": k10, "dims": dims10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.rot90"] = rot90_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.rot90'.")


check_valid('torch.rot90', generated_inputs['torch.rot90'], lib="torch", suffix=0)
