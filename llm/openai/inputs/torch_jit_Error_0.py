
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def jit_error_inputs():
    list_of_inputs = []
    
    input1 = "This is a valid error message."
    input_dict1 = {"msg": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = "Another error message with different content."
    input_dict2 = {"msg": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = "Error message with numbers: 12345"
    input_dict3 = {"msg": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = "Error message with special characters: !@#$%^"
    input_dict4 = {"msg": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = "A very long error message to test the limits." + "a" * 100
    input_dict5 = {"msg": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = "Error message with newline characters: \nThis is a new line."
    input_dict6 = {"msg": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = "Error message in a different language: Mensaje de error válido."
    input_dict7 = {"msg": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = "Error: Division by zero."
    input_dict8 = {"msg": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = "Error: Invalid input type."
    input_dict9 = {"msg": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = "Error: Out of memory."
    input_dict10 = {"msg": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    input11 = "Error: Model not found."
    input_dict11 = {"msg": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["torch.jit.Error"] = jit_error_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.jit.Error' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.jit.Error'.")


check_valid('torch.jit.Error', generated_inputs['torch.jit.Error'], lib="torch", suffix=0)
