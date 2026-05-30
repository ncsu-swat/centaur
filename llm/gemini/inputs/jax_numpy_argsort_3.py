
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, last axis, stable=True
    a = np.array([3.5, 1.2, -0.5, 4.8, -2.3], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, axis 0, descending=True
    a = np.array([[1.5, 2.3], [-0.4, 4.1], [3.2, 0.1]], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 1, stable=True, descending=False
    a = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float16 array, axis 2, descending=True
    a = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis, stable=False
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": False,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 array, axis 3, stable=True
    a = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 3,
        "stable": True,
        "descending": False,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array, descending=True, int32 dtype
    a = np.array([-10.0, 0.0, 10.0, 5.5, -5.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array, axis -1, descending=True
    a = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 array, axis -3, stable=True
    a = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": -3,
        "stable": True,
        "descending": False,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, axis -4, descending=True, int64 dtype
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -4,
        "stable": False,
        "descending": True,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.argsort_3"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_3'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_3'], lib="jax", suffix=3)
