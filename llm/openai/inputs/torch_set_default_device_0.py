
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def set_default_device_inputs():
    list_of_inputs = []
    
    input1 = "cpu"
    input_dict1 = {"device": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = "cuda:0"
    input_dict2 = {"device": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input3 = "cuda"
    input_dict3 = {"device": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = "mps"
    input_dict4 = {"device": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = "cpu:0"
    input_dict5 = {"device": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.set_default_device"] = set_default_device_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_default_device' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_default_device'.")


check_valid('torch.set_default_device', generated_inputs['torch.set_default_device'], lib="torch", suffix=0)
