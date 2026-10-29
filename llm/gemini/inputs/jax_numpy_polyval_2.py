
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyval_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 coefficients with standard unroll
    p = np.array([2.0, 5.0, 1.0], dtype=np.float32)
    x = 3.0
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 2: Linear polynomial with float64
    p = np.array([-1.5, 4.2], dtype=np.float64)
    x = -2.5
    unroll = 1
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 3: Constant polynomial (single coefficient)
    p = np.array([7.0], dtype=np.float32)
    x = 100.0
    unroll = 4
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 4: Higher-degree polynomial with negative and positive coefficients
    p = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    x = 0.5
    unroll = 8
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 5: Zero evaluation point
    p = np.array([3.0, -1.0, 4.0, 2.0], dtype=np.float64)
    x = 0.0
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 6: Integer coefficients with float x
    p = np.array([1, 0, -5, 6], dtype=np.int32)
    x = 2.0
    unroll = 2
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 7: Higher-degree polynomial with large unroll
    p = np.linspace(-1.0, 1.0, 20, dtype=np.float64)
    x = 1.1
    unroll = 32
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 8: Polynomial with all zeros
    p = np.zeros(10, dtype=np.float32)
    x = -10.0
    unroll = 16
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 9: Polynomial with small coefficients
    p = np.array([1e-5, -2e-4, 3e-3, -4e-2], dtype=np.float64)
    x = 10.0
    unroll = 4
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 10: Higher-order polynomial with unroll=64
    p = np.random.RandomState(42).randn(50).astype(np.float32)
    x = 0.99
    unroll = 64
    list_of_inputs.append({"p": p, "x": x, "unroll": unroll})

    # Input 11: Negative evaluation point with unroll=128
    p = np.array([0.5, -0.25, 0.125, -0.0625, 0.03125], dtype=np.float64)
    x = -3.14159
    unroll = 128
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
