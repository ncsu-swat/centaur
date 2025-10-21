
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def is_inference_inputs():
    list_of_inputs = []
    
    input1 = np.array([True])
    list_of_inputs.append(copy.deepcopy({"torch.is_inference": input1}))
    
    return list_of_inputs

generated_inputs["torch.is_inference"] = is_inference_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.is_inference' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.is_inference'.")


check_valid('torch.is_inference', generated_inputs['torch.is_inference'], lib="torch", suffix=0)
