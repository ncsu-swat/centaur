
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def is_autocast_cpu_enabled_inputs():
    list_of_inputs = []

    input1 = True
    input_dict1 = {"torch.is_autocast_cpu_enabled": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = False
    input_dict2 = {"torch.is_autocast_cpu_enabled": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = True
    input_dict3 = {"torch.is_autocast_cpu_enabled": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = False
    input_dict4 = {"torch.is_autocast_cpu_enabled": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = True
    input_dict5 = {"torch.is_autocast_cpu_enabled": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = False
    input_dict6 = {"torch.is_autocast_cpu_enabled": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = True
    input_dict7 = {"torch.is_autocast_cpu_enabled": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = False
    input_dict8 = {"torch.is_autocast_cpu_enabled": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = True
    input_dict9 = {"torch.is_autocast_cpu_enabled": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = False
    input_dict10 = {"torch.is_autocast_cpu_enabled": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.is_autocast_cpu_enabled"] = is_autocast_cpu_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_autocast_cpu_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_autocast_cpu_enabled'.")


check_valid('torch.is_autocast_cpu_enabled', generated_inputs['torch.is_autocast_cpu_enabled'], lib="torch", suffix=0)
