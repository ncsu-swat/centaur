
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def std_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float32'),
        "ddof": 1,
        "keepdims": True,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.randn(3, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": [1],
        "dtype": np.dtype('float64'),
        "ddof": 0,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": [0, 2],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(3, 3).astype(np.float32)
    where = np.array([[True, False, True], [False, True, True], [True, True, False]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.random.randn(4, 3).astype(np.float32)
    mean = np.mean(a, axis=0, keepdims=True)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": None,
        "mean": mean,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.randn(2, 2, 3).astype(np.float32)
    where = np.ones((2, 2, 3), dtype=bool)
    where[0, 0, 0] = False
    input_dict = {
        "a": a,
        "axis": [1, 2],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.random.randn(8).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float64'),
        "ddof": 1,
        "keepdims": False,
        "where": None,
        "mean": None,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": [0],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": None,
        "mean": None,
        "correction": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.random.randn(3, 4).astype(np.float32)
    where = np.ones((3, 4), dtype=bool)
    mean = np.mean(a, axis=1, keepdims=True)
    input_dict = {
        "a": a,
        "axis": [1],
        "dtype": np.dtype('float32'),
        "ddof": 0,
        "keepdims": True,
        "where": where,
        "mean": mean,
        "correction": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.std_3"] = std_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.std_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.std_3'.")


check_valid('jax.numpy.std', generated_inputs['jax.numpy.std_3'], lib="jax", suffix=3)
