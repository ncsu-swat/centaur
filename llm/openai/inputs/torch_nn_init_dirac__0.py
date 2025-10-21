
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def dirac_inputs():
    list_of_inputs = []

    input1 = np.random.rand(6, 6, 6).astype(np.float32)
    offset1 = 3
    input_dict1 = {"tensor": input1, "offset": offset1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(8, 8, 8, 8).astype(np.float64)
    offset2 = 4
    input_dict2 = {"tensor": input2, "offset": offset2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    return list_of_inputs

generated_inputs["torch.nn.init.dirac_"] = dirac_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.init.dirac_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.init.dirac_'.")


check_valid('torch.nn.init.dirac_', generated_inputs['torch.nn.init.dirac_'], lib="torch", suffix=0)
