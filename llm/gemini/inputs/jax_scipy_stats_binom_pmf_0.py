
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def binom_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with int and float32
    k = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    n = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 2: 0D scalar-like arrays
    k = np.array(2, dtype=np.int32)
    n = np.array(10, dtype=np.int32)
    p = np.array(0.3, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 3: 2D arrays with matching shapes
    k = np.array([[1, 2], [3, 4]], dtype=np.int32)
    n = np.array([[10, 10], [10, 10]], dtype=np.int32)
    p = np.array([[0.2, 0.8], [0.5, 0.1]], dtype=np.float32)
    loc = np.array([[0, 1], [-1, 2]], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 4: Broadcasting shapes
    k = np.array([1, 2, 3], dtype=np.int32)
    n = np.array([10], dtype=np.int32)
    p = np.array([0.5], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 5: Boundary probabilities (p = 0.0 and p = 1.0)
    k = np.array([0, 5], dtype=np.int32)
    n = np.array([5, 5], dtype=np.int32)
    p = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 6: Edge case where k > n
    k = np.array([6, 7, 8], dtype=np.int32)
    n = np.array([5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 7: Float32 inputs for all parameters
    k = np.array([1.0, 2.0], dtype=np.float32)
    n = np.array([10.0, 10.0], dtype=np.float32)
    p = np.array([0.25, 0.75], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 8: Float64 inputs
    k = np.array([3.0, 4.0], dtype=np.float64)
    n = np.array([20.0, 20.0], dtype=np.float64)
    p = np.array([0.1, 0.9], dtype=np.float64)
    loc = np.array([1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 9: 3D arrays
    k = np.ones((2, 2, 2), dtype=np.int32) * 2
    n = np.ones((2, 2, 2), dtype=np.int32) * 10
    p = np.ones((2, 2, 2), dtype=np.float32) * 0.4
    loc = np.zeros((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 10: Larger values of n and k
    k = np.array([50, 100], dtype=np.int32)
    n = np.array([100, 200], dtype=np.int32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 11: Negative values of k (results in pmf of 0, which is valid)
    k = np.array([-1, -2], dtype=np.int32)
    n = np.array([10, 10], dtype=np.int32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    return list_of_inputs

generated_inputs["jax.scipy.stats.binom.pmf"] = binom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.binom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.binom.pmf'.")


check_valid('jax.scipy.stats.binom.pmf', generated_inputs['jax.scipy.stats.binom.pmf'], lib="jax", suffix=0)
