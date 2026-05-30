
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D dense grid with int32 dtype
    input_dict = {
        "dimensions": [2, 3],
        "dtype": np.dtype('int32'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic 2D sparse grid with int32 dtype
    input_dict = {
        "dimensions": [2, 3],
        "dtype": np.dtype('int32'),
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D dense grid with int64 dtype
    input_dict = {
        "dimensions": [5],
        "dtype": np.dtype('int64'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D dense grid with int16 dtype
    input_dict = {
        "dimensions": [3, 4, 5],
        "dtype": np.dtype('int16'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D sparse grid with int64 dtype
    input_dict = {
        "dimensions": [3, 4, 5],
        "dtype": np.dtype('int64'),
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Minimal 2D dense grid with int32 dtype
    input_dict = {
        "dimensions": [1, 1],
        "dtype": np.dtype('int32'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger 2D sparse grid with int32 dtype
    input_dict = {
        "dimensions": [10, 20],
        "dtype": np.dtype('int32'),
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D dense grid with int64 dtype
    input_dict = {
        "dimensions": [2, 2, 2, 2],
        "dtype": np.dtype('int64'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D sparse grid with uint32 dtype
    input_dict = {
        "dimensions": [5, 2],
        "dtype": np.dtype('uint32'),
        "sparse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D dense grid with float32 dtype
    input_dict = {
        "dimensions": [8, 4],
        "dtype": np.dtype('float32'),
        "sparse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.indices_2"] = jax_numpy_indices_inputs()

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
