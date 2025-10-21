
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def set_num_threads_inputs():
    list_of_inputs = []

    input_1 = np.int32(1)
    input_dict_1 = {"threads": input_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = np.int64(4)
    input_dict_2 = {"threads": input_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = np.int16(8)
    input_dict_3 = {"threads": input_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = np.int8(16)
    input_dict_4 = {"threads": input_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = np.uint8(32)
    input_dict_5 = {"threads": input_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    input_6 = np.int64(100)
    input_dict_6 = {"threads": input_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = np.int8(64)
    input_dict_7 = {"threads": input_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_8 = np.uint8(1)
    input_dict_8 = {"threads": input_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_9 = np.int32(2)
    input_dict_9 = {"threads": input_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_10 = np.int16(3)
    input_dict_10 = {"threads": input_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    return list_of_inputs

generated_inputs["torch.set_num_threads"] = set_num_threads_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.set_num_threads' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.set_num_threads'.")


check_valid('torch.set_num_threads', generated_inputs['torch.set_num_threads'], lib="torch", suffix=0)
