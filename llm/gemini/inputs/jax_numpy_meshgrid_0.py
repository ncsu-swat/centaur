
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def meshgrid_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default options
    input_dict = {
        "xi": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "copy": True,
        "sparse": False,
        "indexing": "xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array with negative values, sparse=True, indexing='ij'
    input_dict = {
        "xi": np.array([-5, 0, 5, 10], dtype=np.int32),
        "copy": True,
        "sparse": True,
        "indexing": "ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array, sparse=True, indexing='xy'
    input_dict = {
        "xi": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "copy": True,
        "sparse": True,
        "indexing": "xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int64 array, indexing='ij'
    input_dict = {
        "xi": np.array([100, 200], dtype=np.int64),
        "copy": True,
        "sparse": False,
        "indexing": "ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D boolean array
    input_dict = {
        "xi": np.array([True, False], dtype=bool),
        "copy": True,
        "sparse": False,
        "indexing": "xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 1D float32 array
    input_dict = {
        "xi": np.linspace(-10.0, 10.0, 100, dtype=np.float32),
        "copy": True,
        "sparse": True,
        "indexing": "ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D uint8 array
    input_dict = {
        "xi": np.array([0, 128, 255], dtype=np.uint8),
        "copy": True,
        "sparse": False,
        "indexing": "xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element 1D array
    input_dict = {
        "xi": np.array([42.0], dtype=np.float32),
        "copy": True,
        "sparse": True,
        "indexing": "xy"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float16 array
    input_dict = {
        "xi": np.array([-1.5, 1.5], dtype=np.float16),
        "copy": True,
        "sparse": False,
        "indexing": "ij"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int16 array with sequential values
    input_dict = {
        "xi": np.arange(-10, 10, 2, dtype=np.int16),
        "copy": True,
        "sparse": True,
        "indexing": "ij"
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
