
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def bucketize_inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 2.0, 3.0])
    boundaries1 = np.array([1, 3, 5, 7])
    out_int32_1 = False
    right_1 = False
    out1 = np.empty_like(input1, dtype=np.int64)

    input_dict1 = {
        "input": input1,
        "boundaries": boundaries1,
        "out_int32": out_int32_1,
        "right": right_1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1, 2, 3], [4, 5, 6]])
    boundaries2 = np.array([0, 2, 4, 6, 8])
    out_int32_2 = True
    right_2 = True
    out2 = np.empty_like(input2, dtype=np.int32)

    input_dict2 = {
        "input": input2,
        "boundaries": boundaries2,
        "out_int32": out_int32_2,
        "right": right_2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    boundaries3 = np.array([-1, 0, 1])
    out_int32_3 = False
    right_3 = False
    out3 = np.empty_like(input3, dtype=np.int64)

    input_dict3 = {
        "input": input3,
        "boundaries": boundaries3,
        "out_int32": out_int32_3,
        "right": right_3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([5.5, 6.5, 7.5])
    boundaries4 = np.array([5, 7, 9])
    out_int32_4 = False
    right_4 = True
    out4 = np.empty_like(input4, dtype=np.int64)

    input_dict4 = {
        "input": input4,
        "boundaries": boundaries4,
        "out_int32": out_int32_4,
        "right": right_4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([1.0])
    boundaries5 = np.array([0.5, 1.5, 2.5])
    out_int32_5 = True
    right_5 = False
    out5 = np.empty_like(input5, dtype=np.int32)

    input_dict5 = {
        "input": input5,
        "boundaries": boundaries5,
        "out_int32": out_int32_5,
        "right": right_5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([2.0, 4.0, 6.0, 8.0])
    boundaries6 = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    out_int32_6 = False
    right_6 = False
    out6 = np.empty_like(input6, dtype=np.int64)

    input_dict6 = {
        "input": input6,
        "boundaries": boundaries6,
        "out_int32": out_int32_6,
        "right": right_6,
        "out": out6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([10.0, 20.0, 30.0])
    boundaries7 = np.array([5.0, 15.0, 25.0, 35.0])
    out_int32_7 = True
    right_7 = True
    out7 = np.empty_like(input7, dtype=np.int32)

    input_dict7 = {
        "input": input7,
        "boundaries": boundaries7,
        "out_int32": out_int32_7,
        "right": right_7,
        "out": out7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([[0.1, 0.2], [0.3, 0.4]])
    boundaries8 = np.array([0.0, 0.25, 0.5])
    out_int32_8 = False
    right_8 = False
    out8 = np.empty_like(input8, dtype=np.int64)

    input_dict8 = {
        "input": input8,
        "boundaries": boundaries8,
        "out_int32": out_int32_8,
        "right": right_8,
        "out": out8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([1.0, 1.0, 1.0])
    boundaries9 = np.array([0.5, 1.5])
    out_int32_9 = False
    right_9 = True
    out9 = np.empty_like(input9, dtype=np.int64)

    input_dict9 = {
        "input": input9,
        "boundaries": boundaries9,
        "out_int32": out_int32_9,
        "right": right_9,
        "out": out9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input10 = np.array([7.0, 8.0, 9.0])
    boundaries10 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    out_int32_10 = True
    right_10 = False
    out10 = np.empty_like(input10, dtype=np.int32)

    input_dict10 = {
        "input": input10,
        "boundaries": boundaries10,
        "out_int32": out_int32_10,
        "right": right_10,
        "out": out10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.bucketize"] = bucketize_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.bucketize'.")


check_valid('torch.bucketize', generated_inputs['torch.bucketize'], lib="torch", suffix=0)
