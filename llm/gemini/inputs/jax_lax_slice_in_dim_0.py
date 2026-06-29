
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def slice_in_dim_inputs():
    list_of_inputs = []

    # Case 1: 1D float32 array, slice middle, stride 1, axis 0
    operand = np.random.randn(10).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 2,
        'limit_index': 8,
        'stride': 1,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D float32 array, slice axis 0, stride 2
    operand = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 1,
        'limit_index': 7,
        'stride': 2,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D float32 array, slice axis 1, stride 1
    operand = np.random.randn(5, 10).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 0,
        'limit_index': 5,
        'stride': 1,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D float32 array, slice axis 2, stride 3
    operand = np.random.randn(3, 4, 12).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 3,
        'limit_index': 11,
        'stride': 3,
        'axis': 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 4D float32 array, slice axis 3, stride 1
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 1,
        'limit_index': 4,
        'stride': 1,
        'axis': 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 1D float32 array, large, stride 5
    operand = np.random.randn(100).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 10,
        'limit_index': 90,
        'stride': 5,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D int32 array, slice axis 0, stride 2
    operand = np.random.randint(-10, 10, size=(4, 4, 4)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_index': 0,
        'limit_index': 4,
        'stride': 2,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 2D float64 array, slice axis 1, stride 1
    operand = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        'operand': operand,
        'start_index': 2,
        'limit_index': 5,
        'stride': 1,
        'axis': 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 5D float32 array, slice axis 4, stride 2
    operand = np.random.randn(2, 2, 2, 2, 10).astype(np.float32)
    input_dict = {
        'operand': operand,
        'start_index': 2,
        'limit_index': 8,
        'stride': 2,
        'axis': 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 1D int32 array, start 0, limit 1, stride 1
    operand = np.random.randint(-5, 5, size=(1,)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'start_index': 0,
        'limit_index': 1,
        'stride': 1,
        'axis': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.slice_in_dim"] = slice_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_in_dim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_in_dim'.")


check_valid('jax.lax.slice_in_dim', generated_inputs['jax.lax.slice_in_dim'], lib="jax", suffix=0)
