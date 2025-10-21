
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pairwise_distance_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(10, 5).astype(np.float32)
    input2 = np.random.rand(10, 5).astype(np.float32)
    input_dict = {
        "p": 2.0,
        "eps": 1e-6,
        "keepdim": False,
        "input1": torch.tensor(input1),
        "input2": torch.tensor(input2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.PairwiseDistance"] = pairwise_distance_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.PairwiseDistance' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.PairwiseDistance'.")


check_valid('torch.nn.PairwiseDistance', generated_inputs['torch.nn.PairwiseDistance'], lib="torch", suffix=0)
