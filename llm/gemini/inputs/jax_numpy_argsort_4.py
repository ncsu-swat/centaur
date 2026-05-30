
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, standard parameters
    a = np.array([True, False, True, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array, sorting along axis 1, descending
    a = np.array([[True, False, True], [False, False, True]], dtype=bool)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array, axis=0, stable=True, int16 dtype
    a = np.random.choice([True, False], size=(2, 3, 4)).astype(bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D boolean array, stable=False, descending=False, dtype=int32
    a = np.array([False, False, True, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array, axis=0, stable=True, descending=True, dtype=int64
    a = np.random.choice([True, False], size=(5, 5)).astype(bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D boolean array, axis=-2, stable=True, descending=False, dtype=int32
    a = np.random.choice([True, False], size=(2, 2, 3, 3)).astype(bool)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D boolean array (all True), axis=0, descending=True
    a = np.ones(10, dtype=bool)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array (all False), axis=-1, stable=True
    a = np.zeros((4, 4), dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D boolean array, axis=2, stable=True, descending=True, dtype=int16
    a = np.random.choice([True, False], size=(2, 4, 3)).astype(bool)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array, axis=-1, stable=False, descending=True, dtype=int64
    a = np.array([True, True, False, False, True], dtype=bool)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argsort_4"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_4'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_4'], lib="jax", suffix=4)
