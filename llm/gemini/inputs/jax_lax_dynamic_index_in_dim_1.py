
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, keepdims=True, allow_negative_indices=True
    operand = np.arange(5, dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 0,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, keepdims=False, allow_negative_indices=False
    operand = np.arange(10, dtype=np.int32)
    index = np.array(5, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1, negative index, allow_negative_indices=True
    operand = np.arange(12, dtype=np.float32).reshape(3, 4)
    index = np.array(-1, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 1,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=0, keepdims=False, allow_negative_indices=False
    operand = np.random.randn(5, 5).astype(np.float64)
    index = np.array(3, dtype=np.int64)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=2, keepdims=False, allow_negative_indices=True
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    index = np.array(1, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 2,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=1, keepdims=True, negative index
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    index = np.array(-2, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 1,
        'keepdims': True,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, axis=3, keepdims=True, allow_negative_indices=False
    operand = np.ones((2, 3, 2, 5), dtype=np.int32)
    index = np.array(4, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 3,
        'keepdims': True,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D array, axis=2, keepdims=False
    operand = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)
    index = np.array(2, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 2,
        'keepdims': False,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean array as operand
    operand = np.array([[True, False], [False, True]], dtype=bool)
    index = np.array(0, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 1,
        'keepdims': True,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, larger dimensions, axis=0
    operand = np.random.randn(10, 2).astype(np.float32)
    index = np.array(7, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': index,
        'axis': 0,
        'keepdims': False,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_index_in_dim_1"] = dynamic_index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_index_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_index_in_dim_1'.")


check_valid('jax.lax.dynamic_index_in_dim', generated_inputs['jax.lax.dynamic_index_in_dim_1'], lib="jax", suffix=1)
