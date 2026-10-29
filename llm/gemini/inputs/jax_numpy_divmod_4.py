
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def divmod_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    input_dict = {"x1": 10, "x2": np.array([3, 4, 7], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative integer dividend, 1D int64 array
    input_dict = {"x1": -15, "x2": np.array([2, 4, 5, 8], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array
    input_dict = {
        "x1": 100,
        "x2": np.array([[2.5, 4.0], [5.0, 8.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero dividend, 1D int32 array
    input_dict = {"x1": 0, "x2": np.array([1, 2, 3, 4], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer dividend, negative/positive divisors
    input_dict = {"x1": -7, "x2": np.array([-3, -2, 2, 3], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int32 array
    input_dict = {
        "x1": 42,
        "x2": np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D scalar array divisor
    input_dict = {"x1": 17, "x2": np.array(5, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative integer dividend, 2D float64 array
    input_dict = {
        "x1": -50,
        "x2": np.array([[-10.5, 3.2], [-4.1, 7.8]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 array with larger integers
    input_dict = {"x1": 1024, "x2": np.arange(1, 11, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int32 array
    input_dict = {
        "x1": 33,
        "x2": np.full((2, 2, 2, 2), fill_value=5, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D float32 array
    input_dict = {
        "x1": 8,
        "x2": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.divmod_4"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_4'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_4'], lib="jax", suffix=4)
