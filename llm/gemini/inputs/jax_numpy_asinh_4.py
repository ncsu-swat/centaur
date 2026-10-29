
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asinh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive floats
    input_dict = {"x": np.array([0.0, 1.0, 2.5, 10.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of negative floats
    input_dict = {"x": np.array([-0.5, -1.0, -2.0, -100.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array of mixed integers
    input_dict = {"x": np.array([-5, -1, 0, 1, 5, 20], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of floats
    input_dict = {
        "x": np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of floats
    input_dict = {
        "x": np.array(
            [
                [[0.1, -0.2], [0.3, 0.4]],
                [[-0.5, 0.6], [-0.7, 0.8]],
            ],
            dtype=np.float32,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of booleans
    input_dict = {"x": np.array([True, False, True, True], dtype=bool)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of complex numbers
    input_dict = {
        "x": np.array(
            [1.0 + 2.0j, -1.0 - 0.5j, 0.0 + 1.0j, 3.0 + 0.0j],
            dtype=np.complex64,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array of complex numbers
    input_dict = {
        "x": np.array(
            [[1.0j, -1.0j], [2.0 + 3.0j, -4.0 - 5.0j]],
            dtype=np.complex128,
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element 1D array
    input_dict = {"x": np.array([42.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array of integers
    input_dict = {"x": np.array([[0, -10], [100, -1000]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with large values
    input_dict = {
        "x": np.array([1e-4, 1e-2, 1e2, 1e4], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty 1D array
    input_dict = {"x": np.array([], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asinh_4"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_4'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_4'], lib="jax", suffix=4)
