
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array, positive index, keepdims=True
    operand = np.arange(5, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': 2,
        'axis': 0,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, negative index, keepdims=False
    operand = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    input_dict = {
        'operand': operand,
        'index': -1,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, indexing along axis 1, keepdims=True
    operand = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 1,
        'axis': 1,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, indexing along axis 0, keepdims=False, allow_negative_indices=False
    operand = np.random.randint(0, 10, size=(4, 4)).astype(np.int64)
    input_dict = {
        'operand': operand,
        'index': 3,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, indexing along axis 2, keepdims=True
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        'operand': operand,
        'index': 2,
        'axis': 2,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with complex numbers, negative index along axis 1
    operand = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex64)
    input_dict = {
        'operand': operand,
        'index': -2,
        'axis': 1,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, indexing along axis 3, allow_negative_indices=False
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 1,
        'axis': 3,
        'keepdims': True,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean array
    operand = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {
        'operand': operand,
        'index': 3,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, indexing along axis 2
    operand = np.random.randint(-5, 5, size=(2, 3, 1, 4, 5)).astype(np.int32)
    input_dict = {
        'operand': operand,
        'index': 0,
        'axis': 2,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, large negative index, axis 0
    operand = np.random.randn(10, 5).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': -5,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_index_in_dim_2"] = dynamic_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_index_in_dim_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_index_in_dim_2'.")


check_valid('jax.lax.dynamic_index_in_dim', generated_inputs['jax.lax.dynamic_index_in_dim_2'], lib="jax", suffix=2)
