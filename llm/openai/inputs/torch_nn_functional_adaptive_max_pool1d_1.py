
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_max_pool1d_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 5).astype(np.float32)
    output_size1 = 2
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = np.random.rand(1, 5, 7).astype(np.float32)
    output_size2 = 3
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(2, 4, 6).astype(np.float32)
    output_size3 = 1
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 2, 8).astype(np.float32)
    output_size4 = 4
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(3, 1, 10).astype(np.float32)
    output_size5 = 5
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1, 6, 4).astype(np.float32)
    output_size6 = 2
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.rand(2, 5, 5).astype(np.float32)
    output_size7 = 3
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.random.rand(1, 4, 7).astype(np.float32)
    output_size8 = 1
    input_dict8 = {"input": input8, "output_size": output_size8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = np.random.rand(4, 3, 6).astype(np.float32)
    output_size9 = 2
    input_dict9 = {"input": input9, "output_size": output_size9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.random.rand(1, 1, 9).astype(np.float32)
    output_size10 = 4
    input_dict10 = {"input": input10, "output_size": output_size10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.adaptive_max_pool1d_1"] = adaptive_max_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.adaptive_max_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.adaptive_max_pool1d_1'.")


check_valid('torch.nn.functional.adaptive_max_pool1d', generated_inputs['torch.nn.functional.adaptive_max_pool1d_1'], lib="torch", suffix=1)
