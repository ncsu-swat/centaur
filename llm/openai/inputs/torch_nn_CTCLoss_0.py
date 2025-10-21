
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ctcloss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(50, 16, 20).astype(np.float32)
    target1 = np.random.randint(1, 21, size=(16, 10), dtype=np.int64)
    input_lengths1 = np.full((16,), 50, dtype=np.int64)
    target_lengths1 = np.random.randint(1, 11, size=(16,), dtype=np.int64)
    input_dict1 = {
        'blank': 0,
        'reduction': 'mean',
        'zero_infinity': False,
        'log_probs': input1,
        'targets': target1,
        'input_lengths': input_lengths1,
        'target_lengths': target_lengths1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.CTCLoss"] = ctcloss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.CTCLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.CTCLoss'.")


check_valid('torch.nn.CTCLoss', generated_inputs['torch.nn.CTCLoss'], lib="torch", suffix=0)
