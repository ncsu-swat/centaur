
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cumulative_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, axis=0, include_initial=False
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.dtype(np.int32),
        "include_initial": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with negative values, axis=0, include_initial=True
    x = np.array([-1.5, 2.0, -0.5, 4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.dtype(np.float32),
        "include_initial": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis=1, include_initial=False
    x = np.array(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]],
        dtype=np.float64,
    )
    input_dict = {
        "x": x,
        "axis": 1,
        "dtype": np.dtype(np.float64),
        "include_initial": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, axis=0, include_initial=True
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.dtype(np.int64),
        "include_initial": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float array with negative axis, axis=-1, include_initial=False
    x = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": -1,
        "dtype": np.dtype(np.float32),
        "include_initial": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis=2, include_initial=True
    x = np.random.uniform(0.5, 1.5, size=(2, 3, 4)).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": 2,
        "dtype": np.dtype(np.float32),
        "include_initial": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array converted to int64 dtype, axis=1, include_initial=False
    x = np.random.randint(1, 5, size=(2, 4, 3)).astype(np.int32)
    input_dict = {
        "x": x,
        "axis": 1,
        "dtype": np.dtype(np.int64),
        "include_initial": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float64 array, axis=0, include_initial=False
    x = np.random.randn(2, 2, 3, 2).astype(np.float64)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.dtype(np.float64),
        "include_initial": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex array, axis=0, include_initial=True
    x = np.array([1 + 2j, 2 - 1j, -0.5 + 0.5j], dtype=np.complex64)
    input_dict = {
        "x": x,
        "axis": 0,
        "dtype": np.dtype(np.complex64),
        "include_initial": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float array with negative axis, axis=-2, include_initial=True
    x = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict = {
        "x": x,
        "axis": -2,
        "dtype": np.dtype(np.float32),
        "include_initial": True,
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
