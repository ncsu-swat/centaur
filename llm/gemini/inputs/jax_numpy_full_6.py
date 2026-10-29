
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: 1D shape with float32
    input_dict = {
        "shape": (5,),
        "fill_value": 3.14,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D shape with int32
    input_dict = {
        "shape": (2, 4),
        "fill_value": -1.0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D shape (scalar) with float64
    input_dict = {"shape": (), "fill_value": 0.0, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D shape with bool dtype
    input_dict = {
        "shape": (2, 3, 2),
        "fill_value": 1.0,
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D shape with float16
    input_dict = {
        "shape": (1, 2, 2, 3),
        "fill_value": -42.5,
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D shape with complex64
    input_dict = {
        "shape": (3, 3),
        "fill_value": 2.718,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Shape containing 0 (empty array)
    input_dict = {
        "shape": (0, 5),
        "fill_value": 100.0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D shape with int64
    input_dict = {
        "shape": (16,),
        "fill_value": 999.0,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D shape with uint8
    input_dict = {
        "shape": (2, 1, 2, 1, 2),
        "fill_value": 255.0,
        "dtype": np.dtype(np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D shape with large negative fill_value and float32
    input_dict = {
        "shape": (4, 1),
        "fill_value": -1e5,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D shape with complex128
    input_dict = {
        "shape": (2, 2, 2),
        "fill_value": -0.5,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_6"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_6'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_6'], lib="jax", suffix=6)
