
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cov_inputs():
    list_of_inputs = []

    # Input 1
    m = np.random.randn(2, 5).astype(np.float32)
    y = np.random.randn(3, 5).astype(np.float32)
    fweights = np.array([1, 2, 1, 1, 1], dtype=np.int32)
    aweights = np.array([1.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    m = np.random.randn(3, 10).astype(np.float64)
    y = np.random.randn(1, 10).astype(np.float64)
    fweights = np.array([1, 1, 2, 1, 1, 2, 1, 1, 1, 1], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    m = np.random.randn(5, 2).astype(np.float32)
    y = np.random.randn(5, 3).astype(np.float32)
    fweights = np.array([2, 1, 1, 2, 1], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    m = np.random.randn(8, 4).astype(np.float64)
    y = np.random.randn(8, 1).astype(np.float64)
    fweights = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    m = np.random.randn(1, 6).astype(np.float32)
    y = np.random.randn(2, 6).astype(np.float32)
    fweights = np.array([2, 2, 2, 2, 2, 2], dtype=np.int32)
    aweights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    m = np.random.randn(4, 4).astype(np.float32)
    y = np.random.randn(4, 4).astype(np.float32)
    fweights = np.array([1, 2, 3, 4], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": True,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    m = np.random.randn(12, 2).astype(np.float64)
    y = np.random.randn(12, 2).astype(np.float64)
    fweights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": False,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    m = np.random.randn(5, 7).astype(np.float32)
    y = np.random.randn(2, 7).astype(np.float32)
    fweights = np.array([2, 1, 2, 1, 2, 1, 2], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": True,
        "ddof": 1,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    m = np.random.randn(6, 3).astype(np.float64)
    y = np.random.randn(6, 2).astype(np.float64)
    fweights = np.array([1, 2, 1, 2, 1, 2], dtype=np.int32)
    aweights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": False,
        "bias": True,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    m = np.random.randn(3, 3).astype(np.float32)
    y = np.random.randn(3, 3).astype(np.float32)
    fweights = np.array([1, 1, 1], dtype=np.int32)
    aweights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "m": m,
        "y": y,
        "rowvar": True,
        "bias": False,
        "ddof": 0,
        "fweights": fweights,
        "aweights": aweights,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cov"] = cov_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cov' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cov'.")


check_valid('jax.numpy.cov', generated_inputs['jax.numpy.cov'], lib="jax", suffix=0)
