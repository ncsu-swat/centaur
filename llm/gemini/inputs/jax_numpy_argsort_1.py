
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard sort
    a = np.array([3.14, -1.5, 0.0, 2.5, -2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, sorting along axis 0, descending
    a = np.array([[10, 2, 5], [1, 8, 3]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, stable sort along axis 2, descending
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with duplicate values to highlight stable sorting
    a = np.array([2, 1, 2, 1, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.int16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, sorting along axis -2
    a = np.random.rand(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": False,
        "descending": False,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, sorting along axis 1, descending
    a = np.array([[1.5, 2.5], [0.5, -0.5], [3.0, 1.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": True,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float16 array, descending unstable sort
    a = np.array([-10.0, 20.0, 0.0, -5.0], dtype=np.float16)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D uint8 array, sorting along axis 1
    a = np.array([[[1, 5], [3, 2]], [[4, 0], [6, 7]]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": False,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array, last axis, standard sort
    a = np.array([[10.5, -3.2, 8.8], [0.0, -1.1, 5.5]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array with wide range of values
    a = np.array([100000, -500000, 0, 300000, -100000], dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argsort_1"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_1'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_1'], lib="jax", suffix=1)
