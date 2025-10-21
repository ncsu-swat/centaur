
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def scatter_inputs():
    list_of_inputs = []
    
    input1 = torch.randn(5).numpy()
    index1 = torch.tensor([0, 1, 2, 0, 1]).numpy()
    src1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    dim1 = 0
    
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "index": index1,
        "src": src1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.scatter"] = scatter_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.scatter'.")


check_valid('torch.scatter', generated_inputs['torch.scatter'], lib="torch", suffix=0)
