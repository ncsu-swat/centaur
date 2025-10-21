
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_unshuffle_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 3, 8, 8).astype(np.float32)
    downscale_factor1 = 2
    input_dict1 = {"input": input1, "downscale_factor": downscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(1, 6, 4, 4).astype(np.float32)
    downscale_factor2 = 2
    input_dict2 = {"input": input2, "downscale_factor": downscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(1, 9, 2, 2).astype(np.float32)
    downscale_factor3 = 1
    input_dict3 = {"input": input3, "downscale_factor": downscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(1, 4, 16, 16).astype(np.float32)
    downscale_factor4 = 4
    input_dict4 = {"input": input4, "downscale_factor": downscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.rand(1, 1, 32, 32).astype(np.float32)
    downscale_factor5 = 2
    input_dict5 = {"input": input5, "downscale_factor": downscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(2, 3, 8, 8).astype(np.float32)
    downscale_factor6 = 2
    input_dict6 = {"input": input6, "downscale_factor": downscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.random.rand(1, 3, 1, 1).astype(np.float32)
    downscale_factor7 = 1
    input_dict7 = {"input": input7, "downscale_factor": downscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.rand(1, 3, 64, 64).astype(np.float32)
    downscale_factor8 = 8
    input_dict8 = {"input": input8, "downscale_factor": downscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    
    return list_of_inputs

generated_inputs["torch.nn.functional.pixel_unshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pixel_unshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_unshuffle'.")


check_valid('torch.nn.functional.pixel_unshuffle', generated_inputs['torch.nn.functional.pixel_unshuffle'], lib="torch", suffix=0)
