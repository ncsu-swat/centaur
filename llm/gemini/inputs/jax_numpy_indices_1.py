
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def indices_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D grid, dense, int32
    input_dict = {
        "dimensions": (2, 3),
        "dtype": np.dtype(np.int32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard 2D grid, sparse, int32
    input_dict = {
        "dimensions": (2, 3),
        "dtype": np.dtype(np.int32),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D grid, dense, int64
    input_dict = {
        "dimensions": (5,),
        "dtype": np.dtype(np.int64),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D grid, dense, float32
    input_dict = {
        "dimensions": (3, 2, 4),
        "dtype": np.dtype(np.float32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D grid, sparse, float64
    input_dict = {
        "dimensions": (2, 4, 3),
        "dtype": np.dtype(np.float64),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D grid, dense, int16
    input_dict = {
        "dimensions": (2, 1, 3, 2),
        "dtype": np.dtype(np.int16),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty dimensions tuple (0-D), dense, int32
    input_dict = {
        "dimensions": (),
        "dtype": np.dtype(np.int32),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero-sized dimension, sparse, uint32
    input_dict = {
        "dimensions": (0, 4),
        "dtype": np.dtype(np.uint32),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element multi-dimensional grid, dense, uint8
    input_dict = {
        "dimensions": (1, 1, 1),
        "dtype": np.dtype(np.uint8),
        "sparse": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D grid, sparse, int64
    input_dict = {
        "dimensions": (10, 20),
        "dtype": np.dtype(np.int64),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D grid, sparse, int32
    input_dict = {
        "dimensions": (2, 1, 2, 1, 2),
        "dtype": np.dtype(np.int32),
        "sparse": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.indices_1"] = indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.indices_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.indices_1'.")


check_valid('jax.numpy.indices', generated_inputs['jax.numpy.indices_1'], lib="jax", suffix=1)
