
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def get_num_threads_inputs():
    list_of_inputs = []

    input1 = None
    input_dict1 = {"torch.get_num_threads": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = 1
    input_dict2 = {"torch.get_num_threads": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = 0
    input_dict3 = {"torch.get_num_threads": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = -1
    input_dict4 = {"torch.get_num_threads": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = 2
    input_dict5 = {"torch.get_num_threads": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = 10
    input_dict6 = {"torch.get_num_threads": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = 100
    input_dict7 = {"torch.get_num_threads": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.int32(5)
    input_dict8 = {"torch.get_num_threads": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.int64(-5)
    input_dict9 = {"torch.get_num_threads": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.uint8(1)
    input_dict10 = {"torch.get_num_threads": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    input11 = np.int16(20)
    input_dict11 = {"torch.get_num_threads": input11}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    return list_of_inputs

generated_inputs["torch.get_num_threads"] = get_num_threads_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.get_num_threads' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.get_num_threads'.")


check_valid('torch.get_num_threads', generated_inputs['torch.get_num_threads'], lib="torch", suffix=0)
