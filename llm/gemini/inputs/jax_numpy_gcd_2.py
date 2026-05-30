
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gcd_inputs():
    list_of_inputs = []

    # Input 1: Simple scalars (int32)
    x1 = np.int32(12)
    x2 = np.int32(18)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: Negative scalar values (int64)
    x1 = np.int64(-24)
    x2 = np.int64(36)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: Zero values (int16)
    x1 = np.int16(0)
    x2 = np.int16(15)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D arrays of int32
    x1 = np.array([12, 18, 24], dtype=np.int32)
    x2 = np.array([5, 10, 15], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 1D array and a scalar (broadcasting)
    x1 = np.array([12, 24, 36], dtype=np.int32)
    x2 = np.int32(6)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 2D arrays of int64
    x1 = np.array([[12, 24], [36, 48]], dtype=np.int64)
    x2 = np.array([[6, 8], [12, 16]], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: Broadcasting 2D and 1D arrays (int32)
    x1 = np.array([[10, 20], [30, 40]], dtype=np.int32)
    x2 = np.array([5, 10], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 3D arrays (int16)
    x1 = np.array([[[15, 30], [45, 60]]], dtype=np.int16)
    x2 = np.array([[[5, 10], [15, 20]]], dtype=np.int16)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: Large integers (int64)
    x1 = np.array([100000000, 200000000], dtype=np.int64)
    x2 = np.array([50000000, 150000000], dtype=np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: Negative values in 1D arrays (int32)
    x1 = np.array([-10, -20, 30], dtype=np.int32)
    x2 = np.array([5, -10, 15], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.gcd_2"] = gcd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.gcd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.gcd_2'.")


check_valid('jax.numpy.gcd', generated_inputs['jax.numpy.gcd_2'], lib="jax", suffix=2)
