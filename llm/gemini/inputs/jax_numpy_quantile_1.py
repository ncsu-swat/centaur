
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def quantile_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 0,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': False,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis 1
    a = np.random.randn(3, 4).astype(np.float32)
    q = np.array([0.1, 0.9], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, (3, 4)).astype(np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 1,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': True,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, axis 2, float64
    a = np.random.randn(2, 3, 4).astype(np.float64)
    q = np.array([0.5], dtype=np.float64)
    weights = np.random.uniform(0.1, 1.0, (2, 3, 4)).astype(np.float64)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 2,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': False,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values, 1D array
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    q = np.array([0.0, 1.0], dtype=np.float32)
    weights = np.array([0.5, 1.5, 1.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 0,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': True,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with float64, axis 0
    a = np.random.randn(5, 5).astype(np.float64)
    q = np.array([0.3, 0.7], dtype=np.float64)
    weights = np.random.uniform(0.5, 2.0, (5, 5)).astype(np.float64)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 0,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': False,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, negative axis
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    q = np.array([0.5], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, (2, 2, 2, 2)).astype(np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': -1,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': True,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, 0D (scalar) q
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    q = np.array(0.5, dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 0,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': False,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zeros array
    a = np.zeros((2, 3), dtype=np.float32)
    q = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    weights = np.ones((2, 3), dtype=np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': -2,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': True,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with singleton dimension
    a = np.random.randn(3, 1, 3).astype(np.float32)
    q = np.array([0.5], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, (3, 1, 3)).astype(np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 1,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': False,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, large positive values
    a = np.random.uniform(100.0, 1000.0, (4, 2)).astype(np.float32)
    q = np.array([0.99], dtype=np.float32)
    weights = np.random.uniform(1.0, 10.0, (4, 2)).astype(np.float32)
    input_dict = {
        'a': a,
        'q': q,
        'axis': 0,
        'overwrite_input': False,
        'method': 'inverted_cdf',
        'keepdims': True,
        'weights': weights
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.quantile_1"] = quantile_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.quantile_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.quantile_1'.")


check_valid('jax.numpy.quantile', generated_inputs['jax.numpy.quantile_1'], lib="jax", suffix=1)
