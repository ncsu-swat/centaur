
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_full_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D shape with positive fill value, int32 dtype
    input_dict = {
        "shape": (2, 3),
        "fill_value": 5,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D shape with negative fill value, int64 dtype
    input_dict = {
        "shape": (4,),
        "fill_value": -1,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D shape, zero fill value, float32 dtype
    input_dict = {
        "shape": (3, 3, 3),
        "fill_value": 0,
        "dtype": np.dtype('float32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D shape, large positive fill value, float64 dtype
    input_dict = {
        "shape": (1, 5),
        "fill_value": 42,
        "dtype": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D shape, fill value 100, int16 dtype
    input_dict = {
        "shape": (10,),
        "fill_value": 100,
        "dtype": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D shape, large negative fill value, int32 dtype
    input_dict = {
        "shape": (2, 2, 2, 2),
        "fill_value": -999,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D shape (scalar), fill value 7, int32 dtype
    input_dict = {
        "shape": (),
        "fill_value": 7,
        "dtype": np.dtype('int32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D column vector shape, fill value 12345, int64 dtype
    input_dict = {
        "shape": (5, 1),
        "fill_value": 12345,
        "dtype": np.dtype('int64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D shape, fill value 1, bool dtype
    input_dict = {
        "shape": (8, 8),
        "fill_value": 1,
        "dtype": np.dtype('bool')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D shape, fill value -5, int8 dtype
    input_dict = {
        "shape": (2, 4, 3),
        "fill_value": -5,
        "dtype": np.dtype('int8')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.full_5"] = jax_numpy_full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_5'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_5'], lib="jax", suffix=5)
