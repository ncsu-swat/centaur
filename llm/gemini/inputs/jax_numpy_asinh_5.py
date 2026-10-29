
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 values
    input_dict = {"x": np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with wider range
    input_dict = {
        "x": np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of float32
    input_dict = {
        "x": np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of integer values
    input_dict = {"x": np.array([-10, -1, 0, 1, 10], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element 1D array
    input_dict = {"x": np.array([0.5], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of floats
    input_dict = {
        "x": np.linspace(-3.0, 3.0, 8, dtype=np.float32).reshape(2, 2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive and negative values
    input_dict = {
        "x": np.array([-1e4, -1e2, 0.0, 1e2, 1e4], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small values near zero
    input_dict = {
        "x": np.array([-1e-4, -1e-6, 0.0, 1e-6, 1e-4], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive values only
    input_dict = {"x": np.array([0.1, 0.5, 1.0, 2.5, 10.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative values only
    input_dict = {
        "x": np.array([-10.0, -2.5, -1.0, -0.5, -0.1], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D column vector
    input_dict = {"x": np.array([[-2.5], [0.0], [2.5]], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asinh_5"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_5'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_5'], lib="jax", suffix=5)
