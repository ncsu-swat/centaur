
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pixel_shuffle_inputs():
    list_of_inputs = []

    upscale_factor1 = 2
    input1 = torch.randn(1, 4, 4, 4).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor1, "input": input1})

    upscale_factor2 = 3
    input2 = torch.randn(1, 9, 4, 4).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor2, "input": input2})

    upscale_factor3 = 4
    input3 = torch.randn(1, 16, 3, 3).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor3, "input": input3})

    upscale_factor4 = 2
    input4 = torch.randn(2, 4, 8, 8).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor4, "input": input4})

    upscale_factor5 = 3
    input5 = torch.randn(4, 9, 2, 2).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor5, "input": input5})

    upscale_factor6 = 5
    input6 = torch.randn(1, 25, 5, 5).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor6, "input": input6})
    
    upscale_factor7 = 2
    input7 = torch.randn(1, 4, 16, 16).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor7, "input": input7})

    upscale_factor8 = 3
    input8 = torch.randn(2, 9, 8, 8).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor8, "input": input8})

    upscale_factor9 = 4
    input9 = torch.randn(3, 16, 6, 6).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor9, "input": input9})

    upscale_factor10 = 2
    input10 = torch.randn(1, 4, 32, 32).numpy()
    list_of_inputs.append({"upscale_factor": upscale_factor10, "input": input10})

    return list_of_inputs

generated_inputs["torch.nn.PixelShuffle"] = pixel_shuffle_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PixelShuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PixelShuffle'.")


check_valid('torch.nn.PixelShuffle', generated_inputs['torch.nn.PixelShuffle'], lib="torch", suffix=0)
