
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bessel_y0_inputs():
    list_of_inputs = []
    
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"x": input1}
    list_of_inputs.append({"torch.special.bessel_y0": input_dict1})
    
    input2 = np.array([-1.0, -2.0, -3.0])
    input_dict2 = {"x": input2}
    list_of_inputs.append({"torch.special.bessel_y0": input_dict2})
    
    input3 = np.array([0.0, 1.0, 2.0])
    input_dict3 = {"x": input3}
    list_of_inputs.append({"torch.special.bessel_y0": input_dict3})
    
    return list_of_inputs

generated_inputs["torch.special.bessel_y0"] = bessel_y0_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.special.bessel_y0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.special.bessel_y0'.")


check_valid('torch.special.bessel_y0', generated_inputs['torch.special.bessel_y0'], lib="torch", suffix=0)
