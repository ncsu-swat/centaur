
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ravel_multi_index_inputs():
    list_of_inputs = []

    # Input 1: 2D indices, within bounds, C-order, raise
    input_dict = {
        "multi_index": np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32),
        "dims": [2, 3],
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D indices, within bounds, F-order, raise
    input_dict = {
        "multi_index": np.array([[0, 1], [2, 1]], dtype=np.int64),
        "dims": [2, 3],
        "mode": "raise",
        "order": "F",
        "dtype": np.dtype("int64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D indices, within bounds, C-order, raise
    input_dict = {
        "multi_index": np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32),
        "dims": [2, 3, 4],
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D indices, within bounds, C-order, raise
    input_dict = {
        "multi_index": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "dims": [5],
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D indices, out of bounds, clip, C-order
    input_dict = {
        "multi_index": np.array([[-1, 2], [3, -1]], dtype=np.int32),
        "dims": [2, 3],
        "mode": "clip",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D indices, out of bounds, wrap, F-order
    input_dict = {
        "multi_index": np.array([[-1, 2], [3, -1]], dtype=np.int64),
        "dims": [2, 3],
        "mode": "wrap",
        "order": "F",
        "dtype": np.dtype("int64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D indices, out of bounds, ignore, C-order
    input_dict = {
        "multi_index": np.array([[0, 5], [0, 5], [0, 5]], dtype=np.int32),
        "dims": [2, 2, 2],
        "mode": "ignore",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D indices, within bounds, C-order, raise
    input_dict = {
        "multi_index": np.array([[0], [1], [2], [3]], dtype=np.int32),
        "dims": [1, 2, 3, 4],
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D indices with 2D tensor index, clip, C-order
    input_dict = {
        "multi_index": np.array([[[0, 1], [1, 0]], [[1, 2], [0, 1]]], dtype=np.int32),
        "dims": [2, 3],
        "mode": "clip",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D indices with 3D tensor index, raise, C-order
    input_dict = {
        "multi_index": np.ones((2, 2, 2, 2), dtype=np.int32),
        "dims": [3, 3],
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ravel_multi_index_2"] = ravel_multi_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel_multi_index_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel_multi_index_2'.")


check_valid('jax.numpy.ravel_multi_index', generated_inputs['jax.numpy.ravel_multi_index_2'], lib="jax", suffix=2)
