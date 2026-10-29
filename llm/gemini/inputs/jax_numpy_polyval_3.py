
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with positive integer x
    p = np.array([2, 5, 1], dtype=np.int32)
    x = 3
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 2: Float coefficients with negative integer x
    p = np.array([1.5, -2.0, 3.25], dtype=np.float32)
    x = -2
    unroll = 8
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 3: Constant polynomial (length 1)
    p = np.array([42], dtype=np.int64)
    x = 0
    unroll = 1
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 4: Higher degree polynomial with unroll=32
    p = np.linspace(-10, 10, 25, dtype=np.float64)
    x = 1
    unroll = 32
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 5: All zeros coefficients
    p = np.zeros(10, dtype=np.int32)
    x = 5
    unroll = 4
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 6: Large degree polynomial with large unroll
    p = np.ones(50, dtype=np.float32)
    x = -1
    unroll = 64
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 7: Complex coefficients
    p = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    x = 2
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 8: Negative integer coefficients with negative evaluation point
    p = np.array([-5, -3, -1], dtype=np.int64)
    x = -10
    unroll = 2
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 9: Two-element linear polynomial
    p = np.array([3.5, -7.0], dtype=np.float64)
    x = 100
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 10: High degree polynomial with unroll=128
    p = np.arange(1, 17, dtype=np.float32)
    x = -3
    unroll = 128
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    return list_of_inputs


generated_inputs["jax.numpy.polyval_3"] = polyval_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyval_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyval_3'.")


check_valid('jax.numpy.polyval', generated_inputs['jax.numpy.polyval_3'], lib="jax", suffix=3)
