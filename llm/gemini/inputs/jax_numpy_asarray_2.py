
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: 1D integer list with int32 dtype
    input_dict = {
        "a": [1, 2, 3, 4, 5],
        "dtype": np.dtype(np.int32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float list with negative values and float32 dtype
    input_dict = {
        "a": [-2.5, -1.0, 0.0, 1.5, 3.75],
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer list with int64 dtype
    input_dict = {
        "a": [[1, -2, 3], [-4, 5, -6]],
        "dtype": np.dtype(np.int64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float list with float64 dtype
    input_dict = {
        "a": [[0.1, 0.2], [0.3, 0.4], [-0.5, -0.6]],
        "dtype": np.dtype(np.float64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D integer list with int16 dtype
    input_dict = {
        "a": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        "dtype": np.dtype(np.int16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D boolean list with bool dtype
    input_dict = {
        "a": [True, False, True, True, False],
        "dtype": np.dtype(np.bool_),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D boolean list with uint8 dtype
    input_dict = {
        "a": [[True, False], [False, True]],
        "dtype": np.dtype(np.uint8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex list with complex64 dtype
    input_dict = {
        "a": [1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j],
        "dtype": np.dtype(np.complex64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D nested list with float16 dtype
    input_dict = {
        "a": [[[[1.0, -1.0]], [[2.0, -2.0]]]],
        "dtype": np.dtype(np.float16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex list with complex128 dtype
    input_dict = {
        "a": [[1.5 - 0.5j, 2.0 + 1.0j], [-3.0 - 2.5j, 4.2 + 0.0j]],
        "dtype": np.dtype(np.complex128),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D float list with negative values and float32 dtype
    input_dict = {
        "a": [[[-1.0, 2.0], [3.0, -4.0]], [[-5.0, 6.0], [7.0, -8.0]]],
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asarray_2"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_2'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_2'], lib="jax", suffix=2)
