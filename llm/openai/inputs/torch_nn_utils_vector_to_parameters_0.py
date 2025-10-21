
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def vector_to_parameters_inputs():
    list_of_inputs = []
    
    vector1 = np.array([1.0, 2.0, 3.0])
    parameters1 = [np.array([0.1, 0.2, 0.3])]
    list_of_inputs.append({"vector": vector1, "parameters": parameters1})
    
    return list_of_inputs

generated_inputs["torch.nn.utils.vector_to_parameters"] = vector_to_parameters_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.vector_to_parameters' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.vector_to_parameters'.")


check_valid('torch.nn.utils.vector_to_parameters', generated_inputs['torch.nn.utils.vector_to_parameters'], lib="torch", suffix=0)
