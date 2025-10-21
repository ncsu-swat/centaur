
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def is_deterministic_algorithms_warn_only_enabled_inputs():
    list_of_inputs = []

    input1 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input1}))

    input2 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input2}))

    input3 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input3}))

    input4 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input4}))

    input5 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input5}))

    input6 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input6}))

    input7 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input7}))
    
    input8 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input8}))
    
    input9 = True
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input9}))

    input10 = False
    list_of_inputs.append(copy.deepcopy({"torch.is_deterministic_algorithms_warn_only_enabled": input10}))

    return list_of_inputs

generated_inputs["torch.is_deterministic_algorithms_warn_only_enabled"] = is_deterministic_algorithms_warn_only_enabled_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_deterministic_algorithms_warn_only_enabled' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_deterministic_algorithms_warn_only_enabled'.")


check_valid('torch.is_deterministic_algorithms_warn_only_enabled', generated_inputs['torch.is_deterministic_algorithms_warn_only_enabled'], lib="torch", suffix=0)
