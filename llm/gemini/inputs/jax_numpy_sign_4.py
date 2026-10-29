
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sign_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers (positive, negative, zero)
    input_dict = {"x": np.array([0, -3, 7, -10, 15], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of floats
    input_dict = {"x": np.array([-2.5, 0.0, 3.14, -0.001, 100.2], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of integers
    input_dict = {"x": np.array([[1, -1, 0], [-5, 8, -12]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of floats
    input_dict = {
        "x": np.array([[-0.5, 1.2], [0.0, -9.9], [4.4, -3.3]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of integers
    input_dict = {
        "x": np.array([[[1, -2], [3, -4]], [[-5, 6], [0, -8]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of complex numbers
    input_dict = {
        "x": np.array(
            [1 + 0j, 3 + 4j, 0 + 5j, -2 - 2j, 0 + 0j], dtype=np.complex64
        )
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array of complex numbers
    input_dict = {
        "x": np.array([[1 + 1j, -1 - 1j], [0j, 3 - 4j]], dtype=np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of int16
    input_dict = {"x": np.array([-32000, 0, 32000, -1, 1], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with special float values (inf, -inf, nan)
    input_dict = {
        "x": np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with single element
    input_dict = {"x": np.array([-42.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array of random floats
    input_dict = {"x": np.random.randn(2, 3, 4, 5).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sign_4"] = sign_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sign_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sign_4'.")


check_valid('jax.numpy.sign', generated_inputs['jax.numpy.sign_4'], lib="jax", suffix=4)
