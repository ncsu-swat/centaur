
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumulative_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, axis=0, include_initial=False
    input_dict = {
        "x": np.array([1, -2, 3, -4], dtype=np.int32),
        "axis": 0,
        "dtype": np.dtype('int32'),
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, axis=1, include_initial=True
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "axis": 1,
        "dtype": np.dtype('float32'),
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis=0, include_initial=False
    input_dict = {
        "x": np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64),
        "axis": 0,
        "dtype": np.dtype('float64'),
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int64 array, axis=2, include_initial=True
    input_dict = {
        "x": np.arange(1, 9, dtype=np.int64).reshape((2, 2, 2)),
        "axis": 2,
        "dtype": np.dtype('int64'),
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis, include_initial=False
    input_dict = {
        "x": np.random.uniform(-1.0, 1.0, size=(3, 3, 3)).astype(np.float32),
        "axis": -1,
        "dtype": np.dtype('float32'),
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 array, axis=1, include_initial=True
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float64),
        "axis": 1,
        "dtype": np.dtype('float64'),
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array, negative axis, include_initial=True
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "axis": -1,
        "dtype": np.dtype('float32'),
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D int32 array, axis=-2, include_initial=False
    input_dict = {
        "x": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "axis": -2,
        "dtype": np.dtype('int32'),
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, axis=0, include_initial=True
    input_dict = {
        "x": np.ones((2, 3, 4), dtype=np.float32) * 2.0,
        "axis": 0,
        "dtype": np.dtype('float32'),
        "include_initial": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int32 array, axis=3, include_initial=False
    input_dict = {
        "x": np.ones((2, 2, 2, 2, 2), dtype=np.int32),
        "axis": 3,
        "dtype": np.dtype('int32'),
        "include_initial": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cumulative_prod"] = cumulative_prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumulative_prod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumulative_prod'.")


check_valid('jax.numpy.cumulative_prod', generated_inputs['jax.numpy.cumulative_prod'], lib="jax", suffix=0)
