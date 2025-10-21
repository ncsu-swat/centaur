
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def ctc_loss_inputs():
    list_of_inputs = []
    
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    targets1 = np.array([[1, 2, 3]], dtype=np.int64)
    input_lengths1 = np.array([4, 4], dtype=np.int64)
    target_lengths1 = np.array([3], dtype=np.int64)
    blank1 = 0
    reduction1 = 'mean'
    zero_infinity1 = False
    
    input_dict1 = {
        'log_probs': input1,
        'targets': targets1,
        'input_lengths': input_lengths1,
        'target_lengths': target_lengths1,
        'blank': blank1,
        'reduction': reduction1,
        'zero_infinity': zero_infinity1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    
    return list_of_inputs

generated_inputs["torch.nn.functional.ctc_loss"] = ctc_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.ctc_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.ctc_loss'.")


check_valid('torch.nn.functional.ctc_loss', generated_inputs['torch.nn.functional.ctc_loss'], lib="torch", suffix=0)
