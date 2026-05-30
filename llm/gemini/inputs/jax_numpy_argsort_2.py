
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic sorting, int32
    a = np.array([5, 2, 9, 1, 5, 6], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, descending, stable=False, int64
    a = np.array([10, -5, 0, 20, 15], dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, int32
    a = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=1, descending, int64
    a = np.array([[1, 4, 2], [9, 0, 3]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": True,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, axis=-1, stable=False, negative integers
    a = np.random.randint(-100, 100, size=(5, 5), dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": False,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=0, descending, int64
    a = np.random.randint(-50, 50, size=(2, 3, 4), dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": True,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis=1, stable=False, int32
    a = np.random.randint(-10, 10, size=(3, 3, 3), dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": False,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=2, descending, stable=True
    a = np.random.randint(0, 100, size=(2, 4, 3), dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis=3, stable=False, descending, int32
    a = np.random.randint(-1000, 1000, size=(2, 2, 3, 3), dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 3,
        "stable": False,
        "descending": True,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, larger size, stable=True, int64
    a = np.arange(100, -100, -5, dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argsort_2"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_2'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_2'], lib="jax", suffix=2)
