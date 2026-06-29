
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def index_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, index 2, axis 0, keepdims True
    operand = np.arange(10, dtype=np.float32)
    input_dict = {
        'operand': operand,
        'index': 2,
        'axis': 0,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, index 0, axis 0, keepdims False
    operand = np.arange(5, dtype=np.int32)
    input_dict = {
        'operand': operand,
        'index': 0,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, index 1, axis 0, keepdims True
    operand = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 1,
        'axis': 0,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, index 3, axis 1, keepdims False
    operand = np.random.randn(2, 5).astype(np.float64)
    input_dict = {
        'operand': operand,
        'index': 3,
        'axis': 1,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64 array, index 0, axis 2, keepdims True
    operand = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int64)
    input_dict = {
        'operand': operand,
        'index': 0,
        'axis': 2,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, index 2, axis 1, keepdims False
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 2,
        'axis': 1,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array, index 1, axis 3, keepdims True
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 1,
        'axis': 3,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D bool array, index 0, axis 0, keepdims False
    operand = np.random.choice([True, False], size=(2, 3, 1, 2))
    input_dict = {
        'operand': operand,
        'index': 0,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float32 array, index 1, axis 4, keepdims True
    operand = np.random.randn(2, 2, 2, 2, 3).astype(np.float32)
    input_dict = {
        'operand': operand,
        'index': 1,
        'axis': 4,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array, index 2, axis 0, keepdims False
    operand = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {
        'operand': operand,
        'index': 2,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.index_in_dim"] = index_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.index_in_dim' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.index_in_dim'.")


check_valid('jax.lax.index_in_dim', generated_inputs['jax.lax.index_in_dim'], lib="jax", suffix=0)
