
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def logdet_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(3, 3)
    list_of_inputs.append({"input": input1})
    
    input2 = np.random.rand(2, 2)
    list_of_inputs.append({"input": input2})
    
    input3 = np.random.rand(4, 4)
    list_of_inputs.append({"input": input3})
    
    return list_of_inputs

generated_inputs["torch.logdet"] = logdet_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.logdet' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logdet'.")


check_valid('torch.logdet', generated_inputs['torch.logdet'], lib="torch", suffix=0)
