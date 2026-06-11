
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conv_inputs():
    list_of_inputs = []

    # Input 1: 1D convolution with VALID padding, stride 1, default precision
    lhs = np.random.randn(2, 3, 10).astype(np.float32)
    rhs = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1],
        'padding': 'VALID',
        'precision': 'default',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D convolution with SAME padding, stride 2, high precision
    lhs = np.random.randn(1, 2, 16).astype(np.float32)
    rhs = np.random.randn(2, 2, 5).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2],
        'padding': 'SAME',
        'precision': 'high',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution with VALID padding, stride 1, highest precision
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float32)
    rhs = np.random.randn(4, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': 'VALID',
        'precision': 'highest',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution with float64 and SAME padding, stride 2
    lhs = np.random.randn(1, 2, 32, 32).astype(np.float64)
    rhs = np.random.randn(2, 2, 5, 5).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 2],
        'padding': 'SAME',
        'precision': 'default',
        'preferred_element_type': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D convolution with SAME padding, stride 1
    lhs = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(2, 2, 3, 3, 3).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1, 1],
        'padding': 'SAME',
        'precision': 'default',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D convolution with negative values, VALID padding
    lhs = np.random.uniform(-1, 1, (4, 3, 14, 14)).astype(np.float32)
    rhs = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 1],
        'padding': 'VALID',
        'precision': 'high',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D convolution with negative values, stride 3
    lhs = np.random.uniform(-2, 2, (3, 2, 20)).astype(np.float32)
    rhs = np.random.uniform(-2, 2, (4, 2, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [3],
        'padding': 'VALID',
        'precision': 'default',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D convolution with non-square kernels and strides
    lhs = np.random.randn(2, 3, 12, 16).astype(np.float32)
    rhs = np.random.randn(4, 3, 3, 5).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [1, 2],
        'padding': 'SAME',
        'precision': 'high',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D convolution with float64, stride 2, VALID padding
    lhs = np.random.randn(2, 1, 6, 6, 6).astype(np.float64)
    rhs = np.random.randn(2, 1, 2, 2, 2).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 2, 2],
        'padding': 'VALID',
        'precision': 'highest',
        'preferred_element_type': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D convolution with 1x1 kernel and high channel count
    lhs = np.random.randn(1, 16, 8, 8).astype(np.float32)
    rhs = np.random.randn(32, 16, 1, 1).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'window_strides': [2, 2],
        'padding': 'SAME',
        'precision': 'default',
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_1"] = conv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_1'.")


check_valid('jax.lax.conv', generated_inputs['jax.lax.conv_1'], lib="jax", suffix=1)
