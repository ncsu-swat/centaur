
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def var_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype('float32'),
        "ddof": None,  # Maps to 'out' in JAX, must be None
        "keepdims": False,  # Maps to 'ddof' in JAX (False = 0)
        "where": np.ones(10, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, computing along axis=1
    a = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": True,  # Maps to 'ddof' in JAX (True = 1)
        "where": np.ones((4, 5), dtype=bool),
        "mean": np.mean(a, axis=1, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with float64 dtype
    a = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.dtype('float64'),
        "ddof": None,
        "keepdims": False,
        "where": np.ones((3, 4, 5), dtype=bool),
        "mean": np.mean(a, axis=2, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array with a boolean 'where' mask
    a = np.random.randn(3, 4).astype(np.float32)
    where = np.random.choice([True, False], size=(3, 4))
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with precomputed 'mean'
    a = np.random.randn(4, 6).astype(np.float32)
    mean = np.mean(a, axis=1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": True,
        "where": np.ones((4, 6), dtype=bool),
        "mean": mean,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with negative values and negative axis indexing
    a = np.random.uniform(-10.0, 10.0, size=(5, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": np.ones((5, 5), dtype=bool),
        "mean": np.mean(a, axis=-1, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array
    a = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": np.ones(8, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": np.ones((2, 3, 4, 5), dtype=bool),
        "mean": np.mean(a, axis=2, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with 'where' mask
    a = np.random.randn(6, 6).astype(np.float32)
    where = np.random.choice([True, False], size=(6, 6))
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": True,
        "where": where,
        "mean": np.mean(a, axis=1, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with specified 'mean'
    a = np.random.randn(15).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.dtype('float32'),
        "ddof": None,
        "keepdims": False,
        "where": np.ones(15, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.var_3"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_3'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_3'], lib="jax", suffix=3)
