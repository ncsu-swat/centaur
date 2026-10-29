
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyint_inputs():
    list_of_inputs = []

    # Input 1: Basic float 1D polynomial with positive constant
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = 4
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 2: Integer polynomial with zero constant
    p = np.array([3, 2, 1], dtype=np.int32)
    m = 1
    k = 0
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 3: Float64 polynomial with negative constant
    p = np.array([6.0, -4.0, 2.0, -1.0], dtype=np.float64)
    m = 1
    k = -5
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 4: Single-element polynomial (constant polynomial)
    p = np.array([5.0], dtype=np.float32)
    m = 1
    k = 2
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 5: Polynomial with zeros
    p = np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float32)
    m = 1
    k = 7
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 6: Higher-degree polynomial
    p = np.array([10.0, -9.0, 8.0, -7.0, 6.0, -5.0], dtype=np.float64)
    m = 1
    k = -10
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 7: Int64 polynomial with positive constant
    p = np.array([24, 12, 6, 0], dtype=np.int64)
    m = 1
    k = 1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 8: Complex64 polynomial
    p = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    m = 1
    k = 3
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 9: Large degree polynomial
    p = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    m = 1
    k = -1
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 10: Two-element linear polynomial
    p = np.array([2.0, 1.0], dtype=np.float64)
    m = 1
    k = 100
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 11: Polynomial with negative coefficients
    p = np.array([-16.0, -8.0, -4.0, -2.0], dtype=np.float32)
    m = 1
    k = -3
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.polyint_1"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_1'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_1'], lib="jax", suffix=1)
