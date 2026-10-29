
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyint_inputs():
    list_of_inputs = []

    # Input 1: Basic quadratic polynomial, m=1
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = np.array([4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 2: Integer coefficients, m=2
    p = np.array([1, 2, 3], dtype=np.int32)
    m = 2
    k = np.array([4, 5], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 3: Float64 coefficients with negative values, m=1
    p = np.array([-1.5, 3.2, 0.0, -7.1], dtype=np.float64)
    m = 1
    k = np.array([0.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 4: Constant polynomial (degree 0), m=1
    p = np.array([5.0], dtype=np.float32)
    m = 1
    k = np.array([-2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 5: Order 3 integration with alternating constants, m=3
    p = np.array([2.0, -4.0, 1.0], dtype=np.float32)
    m = 3
    k = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 6: All-zero coefficients, int64, m=2
    p = np.array([0, 0, 0], dtype=np.int64)
    m = 2
    k = np.array([0, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 7: Complex coefficients, m=1
    p = np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64)
    m = 1
    k = np.array([1.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 8: High degree polynomial, float64, m=2
    p = np.array([10.0, -5.0, 2.5, -1.25, 0.625], dtype=np.float64)
    m = 2
    k = np.array([3.0, -2.5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 9: Negative int32 coefficients, m=1
    p = np.array([-3, 8, -5, 2], dtype=np.int32)
    m = 1
    k = np.array([-10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 10: Cubic polynomial, m=4
    p = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    m = 4
    k = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 11: Linear polynomial, float64, m=3
    p = np.array([-0.5, 1.5], dtype=np.float64)
    m = 3
    k = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    # Input 12: Degree-0 polynomial integrated twice, m=2
    p = np.array([100.0], dtype=np.float64)
    m = 2
    k = np.array([10.0, 20.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"p": p, "m": m, "k": k}))

    return list_of_inputs


generated_inputs["jax.numpy.polyint_3"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_3'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_3'], lib="jax", suffix=3)
