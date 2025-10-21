
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def put_inputs():
    list_of_inputs = []

    input1 = np.array([1, 2, 3, 4, 5])
    index1 = np.array([0, 2, 4])
    source1 = np.array([10, 20, 30])
    accumulate1 = True

    input_dict1 = {
        "input": input1,
        "index": index1,
        "source": source1,
        "accumulate": accumulate1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.put"] = put_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.put' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.put'.")


check_valid('torch.put', generated_inputs['torch.put'], lib="torch", suffix=0)
