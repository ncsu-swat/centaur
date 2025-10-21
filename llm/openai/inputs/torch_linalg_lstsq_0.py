
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lstsq_inputs():
    list_of_inputs = []

    input_1 = torch.randn(2, 3).numpy()
    input_2 = torch.randn(2, 3).numpy()
    rcond_1 = 1e-15
    driver_1 = 'gelsy'
    input_dict_1 = {'A': input_1, 'B': input_2, 'rcond': rcond_1, 'driver': driver_1}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_3 = torch.randn(4, 5).numpy()
    input_4 = torch.randn(4, 2).numpy()
    rcond_2 = 1e-10
    driver_2 = 'gels'
    input_dict_2 = {'A': input_3, 'B': input_4, 'rcond': rcond_2, 'driver': driver_2}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_5 = torch.randn(3, 3).numpy()
    input_6 = torch.randn(3, 1).numpy()
    rcond_3 = 0.0
    driver_3 = 'gelsd'
    input_dict_3 = {'A': input_5, 'B': input_6, 'rcond': rcond_3, 'driver': driver_3}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_7 = torch.randn(1, 4).numpy()
    input_8 = torch.randn(1, 2).numpy()
    rcond_4 = 1.0
    driver_4 = 'gelss'
    input_dict_4 = {'A': input_7, 'B': input_8, 'rcond': rcond_4, 'driver': driver_4}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_9 = torch.randn(5, 2).numpy()
    input_10 = torch.randn(5, 5).numpy()
    rcond_5 = np.finfo(np.float64).eps * max(5, 2)
    driver_5 = None
    input_dict_5 = {'A': input_9, 'B': input_10, 'rcond': rcond_5, 'driver': driver_5}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_11 = torch.randn(2, 2).numpy()
    input_12 = torch.randn(2, 1).numpy()
    rcond_6 = 1e-6
    driver_6 = 'gelsy'
    input_dict_6 = {'A': input_11, 'B': input_12, 'rcond': rcond_6, 'driver': driver_6}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_13 = torch.randn(3, 4).numpy()
    input_14 = torch.randn(3, 3).numpy()
    rcond_7 = 0.1
    driver_7 = 'gelsd'
    input_dict_7 = {'A': input_13, 'B': input_14, 'rcond': rcond_7, 'driver': driver_7}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    input_15 = torch.randn(1, 5).numpy()
    input_16 = torch.randn(1, 2).numpy()
    rcond_8 = np.finfo(np.float32).eps * max(1, 5)
    driver_8 = None
    input_dict_8 = {'A': input_15, 'B': input_16, 'rcond': rcond_8, 'driver': driver_8}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    input_17 = torch.randn(4, 4).numpy()
    input_18 = torch.randn(4, 1).numpy()
    rcond_9 = 1e-3
    driver_9 = 'gels'
    input_dict_9 = {'A': input_17, 'B': input_18, 'rcond': rcond_9, 'driver': driver_9}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    input_19 = torch.randn(2, 5).numpy()
    input_20 = torch.randn(2, 2).numpy()
    rcond_10 = 0.5
    driver_10 = 'gelss'
    input_dict_10 = {'A': input_19, 'B': input_20, 'rcond': rcond_10, 'driver': driver_10}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.linalg.lstsq"] = lstsq_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.linalg.lstsq' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.lstsq'.")


check_valid('torch.linalg.lstsq', generated_inputs['torch.linalg.lstsq'], lib="torch", suffix=0)
