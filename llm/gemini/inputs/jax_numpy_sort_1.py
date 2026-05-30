
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1: 1D array of floats, sort along last axis, stable=True, descending=False
    a = np.array([3.14, -1.5, 0.0, 2.5, -9.8], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of integers, sort along axis 0, stable=True, descending=True
    a = np.array([[5, 2, 9], [1, 8, 3]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of floats, sort along axis 1, stable=False, descending=False
    a = np.random.randn(5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array of integers, sort along axis 2, stable=True, descending=False
    a = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64)
    input_dict = {
        "a": a,
        "axis": 2,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of integers (with duplicates), sort along axis 0, stable=False, descending=True
    a = np.array([10, 20, 10, 30, 20, 40], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": False,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of floats, sort along axis -1, stable=True, descending=True
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -1,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of floats, sort along negative axis -2, stable=False, descending=False
    a = np.random.uniform(-5.0, 5.0, (4, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "stable": False,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array of integers, sort along axis 1, stable=True, descending=True
    a = np.random.randint(0, 10, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of all negative float values, sort along axis 0, stable=True, descending=False
    a = np.array([-10.5, -20.3, -5.5, -100.1], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": 0,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of large random integers, sort along axis 1, stable=False, descending=True
    a = np.random.randint(-1000, 1000, size=(10, 10)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "stable": False,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sort_1"] = jax_numpy_sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_1'.")


check_valid('jax.numpy.sort', generated_inputs['jax.numpy.sort_1'], lib="jax", suffix=1)
