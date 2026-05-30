
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_cbrt_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive integers
    x = np.array([1, 8, 27, 64, 125], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of negative integers
    x = np.array([-1, -8, -27, -64, -125], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D array of positive floats
    x = np.array([1.0, 3.375, 15.625, 42.875], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D array of negative floats
    x = np.array([-1.0, -3.375, -15.625, -42.875], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D array of integers
    x = np.array([[1, 8, 27], [64, 125, 216]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array with mixed positive and negative integers
    x = np.array([[-1, 8, -27], [64, -125, 216]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 3D array of integers
    x = np.array([[[1, 8], [27, 64]], [[125, 216], [343, 512]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Small fractional float values
    x = np.array([0.001, 0.008, 0.027, 0.064], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Mixed zeros, ones, and negative ones
    x = np.array([0, 1, -1, 0, 1], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array representing large values
    x = np.array([1000000, 8000000, 27000000], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.cbrt_4"] = jax_numpy_cbrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cbrt_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cbrt_4'.")


check_valid('jax.numpy.cbrt', generated_inputs['jax.numpy.cbrt_4'], lib="jax", suffix=4)
