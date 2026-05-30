
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ravel_multi_index_inputs():
    list_of_inputs = []

    # Input 1, 2D, raise mode, C order
    input_dict = {
        "multi_index": np.array([[0, 0, 1], [0, 2, 1]], dtype=np.int32),
        "dims": (2, 3),
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, 2D, wrap mode with negative and out-of-bounds indices
    input_dict = {
        "multi_index": np.array([[1, -1, 0], [2, 5, 1]], dtype=np.int64),
        "dims": (3, 4),
        "mode": "wrap",
        "order": "C",
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, 2D, clip mode, F order
    input_dict = {
        "multi_index": np.array([[0, 1], [2, 3]], dtype=np.int32),
        "dims": (2, 4),
        "mode": "clip",
        "order": "F",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, 3D indices
    input_dict = {
        "multi_index": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        "dims": (2, 3, 4),
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, 1D dimension
    input_dict = {
        "multi_index": np.array([[0, 4, 2]], dtype=np.int64),
        "dims": (5,),
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, Clip mode with larger out-of-bounds indices in F order
    input_dict = {
        "multi_index": np.array([[5, 12, -1], [2, 3, 0]], dtype=np.int32),
        "dims": (10, 5),
        "mode": "clip",
        "order": "F",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, 4D dimensions
    input_dict = {
        "multi_index": np.array([[0], [1], [2], [3]], dtype=np.int32),
        "dims": (2, 2, 3, 4),
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, Wrap mode on boundaries
    input_dict = {
        "multi_index": np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32),
        "dims": (3, 5),
        "mode": "wrap",
        "order": "C",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, 3D, F order, raise mode
    input_dict = {
        "multi_index": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32),
        "dims": (2, 2, 2),
        "mode": "raise",
        "order": "F",
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, Ignore mode
    input_dict = {
        "multi_index": np.array([[0, 1], [0, 1]], dtype=np.int64),
        "dims": (2, 2),
        "mode": "ignore",
        "order": "C",
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ravel_multi_index_1"] = ravel_multi_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel_multi_index_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel_multi_index_1'.")


check_valid('jax.numpy.ravel_multi_index', generated_inputs['jax.numpy.ravel_multi_index_1'], lib="jax", suffix=1)
