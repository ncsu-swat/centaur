
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def square_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    input_dict = {"x": np.array([1, -2, 3, -4, 5], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of floats
    input_dict = {"x": np.array([-2.5, 0.0, 3.5, -1.2], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of integers
    input_dict = {"x": np.array([[1, -2], [3, -4]], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array of complex numbers
    input_dict = {"x": np.array([1 + 2j, 3 - 4j, -1j, 0 + 0j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of booleans
    input_dict = {"x": np.array([True, False, True, False], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D single-element array
    input_dict = {"x": np.array([42.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D array
    input_dict = {"x": np.array([], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of floats
    input_dict = {"x": np.ones((2, 3, 4), dtype=np.float32) * -1.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large magnitude integers
    input_dict = {"x": np.array([-1000, 2000, -3000, 4000], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Special float values (inf, -inf, nan)
    input_dict = {
        "x": np.array([np.inf, -np.inf, np.nan, 1e10], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.square_5"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.square_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.square_5'.")


check_valid('jax.numpy.square', generated_inputs['jax.numpy.square_5'], lib="jax", suffix=5)
