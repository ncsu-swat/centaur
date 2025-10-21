
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pad_sequence_inputs():
    list_of_inputs = []

    sequences1 = [np.array([1, 2, 3]), np.array([4, 5])]
    batch_first1 = False
    padding_value1 = 0.0
    enforce_sorted1 = True
    input_dict1 = {
        "sequences": sequences1,
        "batch_first": batch_first1,
        "padding_value": padding_value1,
        "enforce_sorted": enforce_sorted1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pad_sequence"] = pad_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pad_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pad_sequence'.")


check_valid('torch.nn.utils.rnn.pad_sequence', generated_inputs['torch.nn.utils.rnn.pad_sequence'], lib="torch", suffix=0)
