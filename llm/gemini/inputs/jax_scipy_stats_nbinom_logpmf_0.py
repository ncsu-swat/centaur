
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_nbinom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32
    k = np.array([0, 1, 2, 5], dtype=np.float32)
    n = np.array([2, 2, 2, 2], dtype=np.float32)
    p = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 2: 0D arrays (scalars as tensors)
    k = np.array(3.0, dtype=np.float32)
    n = np.array(5.0, dtype=np.float32)
    p = np.array(0.3, dtype=np.float32)
    loc = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 3: 2D arrays, float64
    k = np.array([[1, 2], [3, 4]], dtype=np.float64)
    n = np.array([[10, 15], [20, 25]], dtype=np.float64)
    p = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    loc = np.array([[0, 0], [0, 0]], dtype=np.float64)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 4: Mixed types (k and loc are integer arrays, others float32)
    k = np.array([1, 2, 3], dtype=np.int32)
    n = np.array([4.5, 4.5, 4.5], dtype=np.float32)
    p = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    loc = np.array([0, 1, 2], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 5: Broadcasting, k is (3, 1), n is (1, 4), p is scalar, loc is scalar
    k = np.array([[1], [2], [3]], dtype=np.float32)
    n = np.array([[5, 6, 7, 8]], dtype=np.float32)
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 6: Higher-dimensional (3D arrays)
    k = np.ones((2, 2, 2), dtype=np.float32) * 5
    n = np.ones((2, 2, 2), dtype=np.float32) * 10
    p = np.ones((2, 2, 2), dtype=np.float32) * 0.75
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 7: Very small probabilities
    k = np.array([10, 20, 30], dtype=np.float32)
    n = np.array([1, 1, 1], dtype=np.float32)
    p = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 8: Very large n
    k = np.array([100], dtype=np.float64)
    n = np.array([1e5], dtype=np.float64)
    p = np.array([0.99], dtype=np.float64)
    loc = np.array([10], dtype=np.float64)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 9: Floating point k (PMF evaluation at non-integers)
    k = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    n = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    p = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    # Input 10: Broadcasting p and loc as 1D arrays, k and n as 2D arrays
    k = np.array([[2, 4], [6, 8]], dtype=np.float32)
    n = np.array([[5, 5], [5, 5]], dtype=np.float32)
    p = np.array([0.2, 0.8], dtype=np.float32)
    loc = np.array([1, 2], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "p": p, "loc": loc})

    return list_of_inputs

generated_inputs["jax.scipy.stats.nbinom.logpmf"] = jax_scipy_stats_nbinom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.nbinom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.nbinom.logpmf'.")


check_valid('jax.scipy.stats.nbinom.logpmf', generated_inputs['jax.scipy.stats.nbinom.logpmf'], lib="jax", suffix=0)
