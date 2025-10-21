
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def parameters_to_vector_inputs():
    list_of_inputs = []
    
    params1 = [torch.tensor([1.0, 2.0, 3.0]).numpy()]
    list_of_inputs.append({"parameters": params1})
    
    return list_of_inputs

generated_inputs["torch.nn.utils.parameters_to_vector"] = parameters_to_vector_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.parameters_to_vector' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.parameters_to_vector'.")


check_valid('torch.nn.utils.parameters_to_vector', generated_inputs['torch.nn.utils.parameters_to_vector'], lib="torch", suffix=0)
