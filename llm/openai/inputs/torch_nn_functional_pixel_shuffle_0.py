
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(1, 9, 1, 1).astype(np.float32)
    upscale_factor1 = 3
    
    input2 = np.random.rand(1, 16, 1, 1).astype(np.float32)
    upscale_factor2 = 4
    
    input3 = np.random.rand(1, 4, 2, 2).astype(np.float32)
    upscale_factor3 = 2
    
    input4 = np.random.rand(2, 9, 2, 2).astype(np.float32)
    upscale_factor4 = 3
    
    input5 = np.random.rand(4, 16, 4, 4).astype(np.float32)
    upscale_factor5 = 4
    
    input6 = np.random.rand(1, 25, 1, 1).astype(np.float32)
    upscale_factor6 = 5
    
    input7 = np.random.rand(1, 36, 1, 1).astype(np.float32)
    upscale_factor7 = 6
    
    input8 = np.random.rand(1, 49, 1, 1).astype(np.float32)
    upscale_factor8 = 7
    
    input9 = np.random.rand(1, 64, 1, 1).astype(np.float32)
    upscale_factor9 = 8
    
    input10 = np.random.rand(2, 16, 2, 2).astype(np.float32)
    upscale_factor10 = 4
    
    input_dict1 = {"input": input1, "upscale_factor": upscale_factor1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input_dict2 = {"input": input2, "upscale_factor": upscale_factor2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    input_dict3 = {"input": input3, "upscale_factor": upscale_factor3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input_dict4 = {"input": input4, "upscale_factor": upscale_factor4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input_dict5 = {"input": input5, "upscale_factor": upscale_factor5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input_dict6 = {"input": input6, "upscale_factor": upscale_factor6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {"input": input7, "upscale_factor": upscale_factor7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input_dict8 = {"input": input8, "upscale_factor": upscale_factor8}
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input_dict9 = {"input": input9, "upscale_factor": upscale_factor9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {"input": input10, "upscale_factor": upscale_factor10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.pixel_shuffle"] = pixel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.pixel_shuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.pixel_shuffle'.")


check_valid('torch.nn.functional.pixel_shuffle', generated_inputs['torch.nn.functional.pixel_shuffle'], lib="torch", suffix=0)
