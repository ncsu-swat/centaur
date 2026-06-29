
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_linalg_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 2-norm, axis 0, no keepdims
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    input_dict = {
        'x': x,
        'ord': 2.0,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, 1-norm, axis 1, keepdims True
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': 1.0,
        'axis': 1,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, inf-norm, axis 2, keepdims False
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': float('inf'),
        'axis': 2,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array, 0-norm (sparsity), axis 0, keepdims True
    x = np.array([0.0, 2.0, -3.0, 0.0, 5.0], dtype=np.float64)
    input_dict = {
        'x': x,
        'ord': 0.0,
        'axis': 0,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, -inf-norm, axis -1, keepdims False
    x = np.random.randn(4, 5).astype(np.float64)
    input_dict = {
        'x': x,
        'ord': float('-inf'),
        'axis': -1,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, 3-norm, axis 1, keepdims True
    x = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': 3.0,
        'axis': 1,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, -1-norm, axis 0, keepdims False
    x = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': -1.0,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, 2-norm, axis 3, keepdims True
    x = np.random.randn(2, 3, 2, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': 2.0,
        'axis': 3,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, 1.5-norm, axis 0, keepdims False
    x = np.array([-1.2, 3.4, -5.6], dtype=np.float32)
    input_dict = {
        'x': x,
        'ord': 1.5,
        'axis': 0,
        'keepdims': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, -2-norm, axis 1, keepdims True
    x = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        'x': x,
        'ord': -2.0,
        'axis': 1,
        'keepdims': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_5"] = jax_numpy_linalg_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_5'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_5'], lib="jax", suffix=5)
