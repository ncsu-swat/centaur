
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_unshuffle_inputs():
    list_of_inputs = []
    
    downscale_factor1 = 2
    input1 = torch.randn(1, 4, 8, 8).numpy()
    input_dict1 = {"downscale_factor": downscale_factor1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    downscale_factor2 = 3
    input2 = torch.randn(1, 9, 12, 12).numpy()
    input_dict2 = {"downscale_factor": downscale_factor2, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    downscale_factor3 = 1
    input3 = torch.randn(2, 3, 16, 16).numpy()
    input_dict3 = {"downscale_factor": downscale_factor3, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    downscale_factor4 = 4
    input4 = torch.randn(1, 16, 20, 20).numpy()
    input_dict4 = {"downscale_factor": downscale_factor4, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    downscale_factor5 = 5
    input5 = torch.randn(4, 25, 25, 25).numpy()
    input_dict5 = {"downscale_factor": downscale_factor5, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    downscale_factor6 = 2
    input6 = torch.randn(2, 1, 6, 6).numpy()
    input_dict6 = {"downscale_factor": downscale_factor6, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    downscale_factor7 = 3
    input7 = torch.randn(1, 64, 9, 9).numpy()
    input_dict7 = {"downscale_factor": downscale_factor7, "input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    downscale_factor8 = 4
    input8 = torch.randn(3, 1, 12, 12).numpy()
    input_dict8 = {"downscale_factor": downscale_factor8, "input": input8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    downscale_factor9 = 2
    input9 = torch.randn(1, 1, 4, 4).numpy()
    input_dict9 = {"downscale_factor": downscale_factor9, "input": input9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    downscale_factor10 = 5
    input10 = torch.randn(2, 3, 10, 10).numpy()
    input_dict10 = {"downscale_factor": downscale_factor10, "input": input10}
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["torch.nn.PixelUnshuffle"] = pixel_unshuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PixelUnshuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PixelUnshuffle'.")


check_valid('torch.nn.PixelUnshuffle', generated_inputs['torch.nn.PixelUnshuffle'], lib="torch", suffix=0)
