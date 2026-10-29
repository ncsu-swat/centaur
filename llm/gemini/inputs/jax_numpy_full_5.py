
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: 1D shape with int32 dtype
    input_dict = {
        "shape": (5,),
        "fill_value": 7,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape with float32 dtype and negative integer fill value
    input_dict = {
        "shape": (2, 3),
        "fill_value": -4,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D (scalar) shape with float64 dtype
    input_dict = {
        "shape": (),
        "fill_value": 42,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape with bool dtype and 0 fill value
    input_dict = {
        "shape": (2, 2, 2),
        "fill_value": 0,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D shape with bool dtype and non-zero fill value
    input_dict = {
        "shape": (1, 4, 1),
        "fill_value": 1,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D shape with uint8 dtype
    input_dict = {
        "shape": (1, 2, 3, 4),
        "fill_value": 255,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape containing zero dimension
    input_dict = {
        "shape": (0, 5),
        "fill_value": 10,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D shape with complex64 dtype
    input_dict = {
        "shape": (3, 3),
        "fill_value": -1,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D shape with int16 dtype
    input_dict = {
        "shape": (100,),
        "fill_value": 300,
        "dtype": np.dtype(np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D shape with int8 dtype and negative fill value
    input_dict = {
        "shape": (1, 1, 2, 1, 3),
        "fill_value": -128,
        "dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D shape with uint32 dtype
    input_dict = {
        "shape": (4, 4),
        "fill_value": 65536,
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_5"] = full_inputs()

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
