
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def meshgrid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default cartesian indexing, non-sparse
    input_dict = {
        "xi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "copy": True,
        "sparse": False,
        "indexing": "xy",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array with negative numbers, matrix indexing, sparse
    input_dict = {
        "xi": np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float64),
        "copy": True,
        "sparse": True,
        "indexing": "ij",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array, cartesian indexing, sparse
    input_dict = {
        "xi": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "copy": True,
        "sparse": True,
        "indexing": "xy",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 array with negative values, matrix indexing, non-sparse
    input_dict = {
        "xi": np.array([-10, -5, 0, 5, 10], dtype=np.int64),
        "copy": True,
        "sparse": False,
        "indexing": "ij",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element 1D float32 array
    input_dict = {
        "xi": np.array([42.0], dtype=np.float32),
        "copy": True,
        "sparse": False,
        "indexing": "xy",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element 1D int32 array, sparse, matrix indexing
    input_dict = {
        "xi": np.array([-1], dtype=np.int32),
        "copy": True,
        "sparse": True,
        "indexing": "ij",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with zeros and ones, matrix indexing
    input_dict = {
        "xi": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "copy": True,
        "sparse": False,
        "indexing": "ij",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 array, sparse, cartesian indexing
    input_dict = {
        "xi": np.array([100.5, 200.75, 300.125], dtype=np.float64),
        "copy": True,
        "sparse": True,
        "indexing": "xy",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 array with positive values, matrix indexing, sparse
    input_dict = {
        "xi": np.array([10, 20, 30, 40], dtype=np.int64),
        "copy": True,
        "sparse": True,
        "indexing": "ij",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D float32 array
    input_dict = {
        "xi": np.linspace(-1.0, 1.0, 50, dtype=np.float32),
        "copy": True,
        "sparse": False,
        "indexing": "xy",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.meshgrid"] = meshgrid_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.meshgrid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.meshgrid'.")


check_valid('jax.numpy.meshgrid', generated_inputs['jax.numpy.meshgrid'], lib="jax", suffix=0)
