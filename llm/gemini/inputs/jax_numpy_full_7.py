
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: 1D shape with True, boolean dtype
    input_dict = {
        "shape": (5,),
        "fill_value": True,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape with False, int32 dtype
    input_dict = {
        "shape": (2, 3),
        "fill_value": False,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D shape with True, float32 dtype
    input_dict = {
        "shape": (2, 4, 3),
        "fill_value": True,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar shape with True, float64 dtype
    input_dict = {
        "shape": (),
        "fill_value": True,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D shape with False, uint8 dtype
    input_dict = {
        "shape": (1, 2, 1, 4),
        "fill_value": False,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Shape with a zero dimension, complex64 dtype
    input_dict = {
        "shape": (0, 5),
        "fill_value": True,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D shape with True, int8 dtype
    input_dict = {
        "shape": (2, 1, 2, 1, 3),
        "fill_value": True,
        "dtype": np.dtype(np.int8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D square shape with False, int64 dtype
    input_dict = {
        "shape": (8, 8),
        "fill_value": False,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D shape with True, float16 dtype
    input_dict = {
        "shape": (1024,),
        "fill_value": True,
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D shape with False, complex128 dtype
    input_dict = {
        "shape": (3, 3, 3),
        "fill_value": False,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_7"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_7'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_7'], lib="jax", suffix=7)
