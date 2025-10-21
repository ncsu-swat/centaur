
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def pack_padded_sequence_inputs():
    list_of_inputs = []

    input1 = np.random.rand(10, 5).astype(np.float32)
    lengths1 = np.array([3, 5, 2, 7, 4, 1, 6, 8, 9, 3]).astype(np.int64)
    lengths1.sort()
    lengths1 = lengths1[::-1]
    batch_first1 = True
    enforce_sorted1 = True

    input_dict1 = {
        "input": input1,
        "lengths": lengths1,
        "batch_first": batch_first1,
        "enforce_sorted": enforce_sorted1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(5, 10).astype(np.float32)
    lengths2 = np.array([1, 2, 3, 4, 5]).astype(np.int64)
    lengths2.sort()
    lengths2 = lengths2[::-1]
    batch_first2 = False
    enforce_sorted2 = False

    input_dict2 = {
        "input": input2,
        "lengths": lengths2,
        "batch_first": batch_first2,
        "enforce_sorted": enforce_sorted2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(20, 3).astype(np.float32)
    lengths3 = np.array([5, 10, 3, 7, 2, 8, 1, 6, 9, 4, 5, 3, 7, 10, 2, 6, 8, 1, 4, 5]).astype(np.int64)
    lengths3.sort()
    lengths3 = lengths3[::-1]
    batch_first3 = True
    enforce_sorted3 = False

    input_dict3 = {
        "input": input3,
        "lengths": lengths3,
        "batch_first": batch_first3,
        "enforce_sorted": enforce_sorted3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.nn.utils.rnn.pack_padded_sequence"] = pack_padded_sequence_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.utils.rnn.pack_padded_sequence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.rnn.pack_padded_sequence'.")


check_valid('torch.nn.utils.rnn.pack_padded_sequence', generated_inputs['torch.nn.utils.rnn.pack_padded_sequence'], lib="torch", suffix=0)
