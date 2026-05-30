
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 coefficients with standard float x and default unroll
    p = np.array([2.0, 5.0, 1.0], dtype=np.float32)
    x = 3.0
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 2: Negative and float64 coefficients, negative float x, smaller unroll
    p = np.array([-1.5, 2.5, 0.0, -3.2], dtype=np.float64)
    x = -1.5
    unroll = 8
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 3: Integer coefficients, float x, unroll=4
    p = np.array([1, -2, 3, -4], dtype=np.int32)
    x = 2.0
    unroll = 4
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 4: Float64 coefficients, large float x, larger unroll
    p = np.array([0.5, -0.5], dtype=np.float64)
    x = 10.5
    unroll = 32
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 5: Small float32 coefficients, zero float x, unroll=64
    p = np.array([1e-3, 1e-2, 1e-1, 1.0], dtype=np.float32)
    x = 0.0
    unroll = 64
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 6: Single-element coefficient, arbitrary float x, default unroll
    p = np.array([42.0], dtype=np.float32)
    x = 123.45
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 7: High-degree random float64 polynomial coefficients, fractional float x, unroll=128
    p = np.random.randn(10).astype(np.float64)
    x = 0.5
    unroll = 128
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 8: All-zeros float32 coefficients, negative float x, default unroll
    p = np.zeros(5, dtype=np.float32)
    x = -0.1
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 9: Int64 coefficients, float x, minimal unroll of 1
    p = np.array([3, 2, 1], dtype=np.int64)
    x = 4.5
    unroll = 1
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 10: Higher-order float32 coefficients, x = 1.0, unroll = 256
    p = np.array([1.5, -2.5, 3.5, -4.5, 5.5], dtype=np.float32)
    x = 1.0
    unroll = 256
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    return list_of_inputs

generated_inputs["jax.numpy.polyval_2"] = polyval_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyval_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyval_2'.")


check_valid('jax.numpy.polyval', generated_inputs['jax.numpy.polyval_2'], lib="jax", suffix=2)
