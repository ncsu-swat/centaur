
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def searchsorted_inputs():
    list_of_inputs = []

    input1 = np.array([1, 3, 5, 7, 9])
    values1 = np.array([2, 4, 6])
    out_int32_1 = False
    right_1 = False
    side_1 = None
    out_1 = np.empty(values1.shape, dtype=np.int64)
    sorter_1 = None
    input_dict1 = {
        "sorted_sequence": input1,
        "values": values1,
        "out_int32": out_int32_1,
        "right": right_1,
        "side": side_1,
        "out": out_1,
        "sorter": sorter_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 3, 5], [2, 4, 6]])
    values2 = np.array([[2, 4], [3, 5]])
    out_int32_2 = True
    right_2 = True
    side_2 = 'right'
    out_2 = np.empty(values2.shape, dtype=np.int32)
    sorter_2 = None
    input_dict2 = {
        "sorted_sequence": input2,
        "values": values2,
        "out_int32": out_int32_2,
        "right": right_2,
        "side": side_2,
        "out": out_2,
        "sorter": sorter_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-5, -3, -1, 1, 3])
    values3 = np.array([-4, 0, 2])
    out_int32_3 = False
    right_3 = False
    side_3 = 'left'
    out_3 = np.empty(values3.shape, dtype=np.int64)
    sorter_3 = None
    input_dict3 = {
        "sorted_sequence": input3,
        "values": values3,
        "out_int32": out_int32_3,
        "right": right_3,
        "side": side_3,
        "out": out_3,
        "sorter": sorter_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values4 = np.array([1.5, 2.5, 3.5])
    out_int32_4 = True
    right_4 = False
    side_4 = None
    out_4 = np.empty(values4.shape, dtype=np.int32)
    sorter_4 = None
    input_dict4 = {
        "sorted_sequence": input4,
        "values": values4,
        "out_int32": out_int32_4,
        "right": right_4,
        "side": side_4,
        "out": out_4,
        "sorter": sorter_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([1, 3, 5, 7, 9])
    values5 = np.array([0, 2, 4, 6, 8])
    out_int32_5 = False
    right_5 = True
    side_5 = 'right'
    out_5 = np.empty(values5.shape, dtype=np.int64)
    sorter_5 = None
    input_dict5 = {
        "sorted_sequence": input5,
        "values": values5,
        "out_int32": out_int32_5,
        "right": right_5,
        "side": side_5,
        "out": out_5,
        "sorter": sorter_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.searchsorted"] = searchsorted_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.searchsorted' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.searchsorted'.")


check_valid('torch.searchsorted', generated_inputs['torch.searchsorted'], lib="torch", suffix=0)
