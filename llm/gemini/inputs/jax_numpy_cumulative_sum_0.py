
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumulative_sum_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array of integers
    x = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.int32,
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, accumulation along axis 1 with initial value included
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "axis": 1,
        "dtype": np.float32,
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, negative values, along axis 2
    x = np.random.randn(3, 4, 5).astype(np.float64)
    input_dict = {
        "x": x,
        "axis": 2,
        "dtype": np.float64,
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, negative and positive values, along axis 0 with initial
    x = np.random.randint(-10, 10, size=(5, 5)).astype(np.int64)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.int64,
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, along negative axis -1
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": -1,
        "dtype": np.float32,
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean array, accumulated with integer dtype conversion
    x = np.array([True, False, True, True], dtype=bool)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.int32,
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, using negative axis
    x = np.random.randn(10).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": -1,
        "dtype": np.float32,
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array accumulated as float64, along axis -2
    x = np.random.randn(6, 6).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": -2,
        "dtype": np.float64,
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int32 array, accumulated as int64, along axis 1
    x = np.random.randint(-5, 5, size=(4, 3, 2)).astype(np.int32)
    input_dict = {
        "x": x,
        "axis": 1,
        "dtype": np.int64,
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, along axis 3
    x = np.ones((2, 3, 4, 5), dtype=np.float32)
    input_dict = {
        "x": x,
        "axis": 3,
        "dtype": np.float32,
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cumulative_sum"] = cumulative_sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumulative_sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumulative_sum'.")


check_valid('jax.numpy.cumulative_sum', generated_inputs['jax.numpy.cumulative_sum'], lib="jax", suffix=0)
