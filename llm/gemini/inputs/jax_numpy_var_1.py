
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def var_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with random mask
    a = np.random.randn(4, 5).astype(np.float32)
    where = np.random.choice([True, False], size=(4, 5), p=[0.8, 0.2])
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": np.mean(a, axis=1, keepdims=True),
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array
    a = np.random.randn(5, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float64,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array using 'correction' to define degrees of freedom
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=2, keepdims=True),
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 3,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=3, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with specific mask
    a = np.random.randn(3, 4).astype(np.float32)
    where = np.ones((3, 4), dtype=bool)
    where[0, 0] = False
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": where,
        "mean": np.mean(a, axis=1, keepdims=True),
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with negative axis
    a = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": -1,
        "dtype": np.float64,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=-1, keepdims=True),
        "correction": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D array with float64
    a = np.random.randn(50, 50).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float64,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with correction=3
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32,
        "ddof": None,
        "keepdims": False,
        "where": np.ones_like(a, dtype=bool),
        "mean": np.mean(a, axis=0, keepdims=True),
        "correction": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.var_1"] = var_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.var_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.var_1'.")


check_valid('jax.numpy.var', generated_inputs['jax.numpy.var_1'], lib="jax", suffix=1)
