
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gcd_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays with int32
    x1 = np.array([12, 18, 24], dtype=np.int32)
    x2 = np.array([5, 10, 15], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D arrays containing negative numbers with int32
    x1 = np.array([-12, 18, -24], dtype=np.int32)
    x2 = np.array([5, -10, 15], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Broadcasting a 1D array with a single-element array
    x1 = np.array([12, 18, 24], dtype=np.int32)
    x2 = np.array([6], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 2D arrays (matrices) with int64
    x1 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    x2 = np.array([[5, 15], [25, 35]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: Another 1D array with int64
    x1 = np.array([8, 16, 32], dtype=np.int64)
    x2 = np.array([4, 12, 20], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: Zero values (GCD with zero) with int32
    x1 = np.array([0, 0, 15], dtype=np.int32)
    x2 = np.array([5, 0, 0], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 3D arrays to test higher dimensions with int32
    x1 = np.ones((2, 2, 2), dtype=np.int32) * 12
    x2 = np.ones((2, 2, 2), dtype=np.int32) * 18
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: Broadcasting with shape (3, 1) and (1, 3) with int32
    x1 = np.array([[12], [24], [36]], dtype=np.int32)
    x2 = np.array([[8, 16, 20]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Large values with int64
    x1 = np.array([1000000000, 2000000000], dtype=np.int64)
    x2 = np.array([3000000000, 4000000000], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Prime numbers (resulting in gcd of 1) with int32
    x1 = np.array([13, 17, 19], dtype=np.int32)
    x2 = np.array([23, 29, 31], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 11: 1D array and 0D array (scalar array) with int32
    x1 = np.array([100, 200, 300], dtype=np.int32)
    x2 = np.array(50, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.gcd_1"] = gcd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gcd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gcd_1'.")


check_valid('jax.numpy.gcd', generated_inputs['jax.numpy.gcd_1'], lib="jax", suffix=1)
