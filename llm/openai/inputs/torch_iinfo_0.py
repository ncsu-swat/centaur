
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_iinfo_inputs():
    list_of_inputs = []

    input1 = np.dtype(np.int8)
    input_dict1 = {"dtype": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.dtype(np.int16)
    input_dict2 = {"dtype": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.dtype(np.int32)
    input_dict3 = {"dtype": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.dtype(np.int64)
    input_dict4 = {"dtype": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.dtype(np.uint8)
    input_dict5 = {"dtype": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.dtype(np.uint16)
    input_dict6 = {"dtype": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.iinfo"] = torch_iinfo_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.iinfo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.iinfo'.")


check_valid('torch.iinfo', generated_inputs['torch.iinfo'], lib="torch", suffix=0)
