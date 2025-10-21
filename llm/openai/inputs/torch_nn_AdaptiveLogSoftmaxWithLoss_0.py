
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def adaptive_log_softmax_with_loss_inputs():
    list_of_inputs = []

    input1 = np.random.rand(10, 20).astype(np.float32)
    target1 = np.random.randint(0, 100, size=(10,)).astype(np.int64)
    inputs1 = {
        "in_features": 20,
        "n_classes": 100,
        "cutoffs": [10, 50, 90],
        "div_value": 4.0,
        "head_bias": True,
        "dtype": np.float32,
        "input": input1,
        "target": target1
    }
    list_of_inputs.append(copy.deepcopy(inputs1))

    return list_of_inputs

generated_inputs["torch.nn.AdaptiveLogSoftmaxWithLoss"] = adaptive_log_softmax_with_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AdaptiveLogSoftmaxWithLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AdaptiveLogSoftmaxWithLoss'.")


check_valid('torch.nn.AdaptiveLogSoftmaxWithLoss', generated_inputs['torch.nn.AdaptiveLogSoftmaxWithLoss'], lib="torch", suffix=0)
