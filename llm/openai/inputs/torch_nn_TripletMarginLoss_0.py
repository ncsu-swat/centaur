
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def triplet_margin_loss_inputs():
    list_of_inputs = []
    
    input_dict_1 = {
        'margin': 1.0,
        'p': 2,
        'eps': 1e-06,
        'swap': False,
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'anchor': np.random.rand(10, 128).astype(np.float32),
        'positive': np.random.rand(10, 128).astype(np.float32),
        'negative': np.random.rand(10, 128).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))
    
    input_dict_2 = {
        'margin': 0.5,
        'p': 1,
        'eps': 1e-07,
        'swap': True,
        'size_average': False,
        'reduce': False,
        'reduction': 'none',
        'anchor': np.random.rand(5, 64).astype(np.float32),
        'positive': np.random.rand(5, 64).astype(np.float32),
        'negative': np.random.rand(5, 64).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))
    
    input_dict_3 = {
        'margin': 1.0,
        'p': 3,
        'eps': 1e-08,
        'swap': False,
        'size_average': True,
        'reduce': True,
        'reduction': 'sum',
        'anchor': np.random.rand(20, 32).astype(np.float32),
        'positive': np.random.rand(20, 32).astype(np.float32),
        'negative': np.random.rand(20, 32).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    input_dict_4 = {
        'margin': 2.0,
        'p': 2,
        'eps': 1e-05,
        'swap': True,
        'size_average': False,
        'reduce': False,
        'reduction': 'none',
        'anchor': np.random.rand(1, 128).astype(np.float32),
        'positive': np.random.rand(1, 128).astype(np.float32),
        'negative': np.random.rand(1, 128).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        'margin': 0.1,
        'p': 4,
        'eps': 1e-09,
        'swap': False,
        'size_average': True,
        'reduce': True,
        'reduction': 'mean',
        'anchor': np.random.rand(15, 64).astype(np.float32),
        'positive': np.random.rand(15, 64).astype(np.float32),
        'negative': np.random.rand(15, 64).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["torch.nn.TripletMarginLoss"] = triplet_margin_loss_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.TripletMarginLoss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.TripletMarginLoss'.")


check_valid('torch.nn.TripletMarginLoss', generated_inputs['torch.nn.TripletMarginLoss'], lib="torch", suffix=0)
