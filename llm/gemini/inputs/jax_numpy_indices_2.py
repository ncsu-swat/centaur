
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def indices_inputs():
    list_of_inputs = []

    # Input 1: 1D grid, default-like int32, dense
    input_dict = {
        "dimensions": [5],
        "dtype": np.dtype(np.int32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D grid, int64, sparse
    input_dict = {
        "dimensions": [2, 3],
        "dtype": np.dtype(np.int64),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D grid, float32, dense
    input_dict = {
        "dimensions": [2, 3, 4],
        "dtype": np.dtype(np.float32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D grid, float64, sparse
    input_dict = {
        "dimensions": [1, 2, 3, 2],
        "dtype": np.dtype(np.float64),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D grid (empty list of dimensions), int32, dense
    input_dict = {"dimensions": [], "dtype": np.dtype(np.int32), "sparse": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D grid, int64, sparse
    input_dict = {"dimensions": [], "dtype": np.dtype(np.int64), "sparse": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Grid with dimension of size 0, int32, dense
    input_dict = {
        "dimensions": [0, 4],
        "dtype": np.dtype(np.int32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Grid with dimension of size 0, int64, sparse
    input_dict = {
        "dimensions": [3, 0],
        "dtype": np.dtype(np.int64),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element multi-dimensional grid (all 1s), uint32, dense
    input_dict = {
        "dimensions": [1, 1, 1],
        "dtype": np.dtype(np.uint32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D square grid, int16, sparse
    input_dict = {
        "dimensions": [8, 8],
        "dtype": np.dtype(np.int16),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D grid, int32, sparse
    input_dict = {
        "dimensions": [2, 1, 2, 1, 2],
        "dtype": np.dtype(np.int32),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Large 1D grid, uint64, dense
    input_dict = {
        "dimensions": [100],
        "dtype": np.dtype(np.uint64),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.indices_2"] = indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.indices_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.indices_2'.")


check_valid('jax.numpy.indices', generated_inputs['jax.numpy.indices_2'], lib="jax", suffix=2)
