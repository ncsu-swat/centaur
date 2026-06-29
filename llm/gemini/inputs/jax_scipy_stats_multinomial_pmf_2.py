
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_multinomial_pmf_inputs():
    list_of_inputs = []

    # Input 1: 1D case, 3 categories, scalar n
    x = np.array([1, 2, 3], dtype=np.int32)
    n = np.array(6, dtype=np.int32)
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 2: 1D case, alternative counts, scalar n
    x = np.array([0, 5, 0], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([0.1, 0.8, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 3: Batch of samples (2D x, scalar n, 1D p)
    x = np.array([[1, 2], [2, 1]], dtype=np.int32)
    n = np.array(3, dtype=np.int32)
    p = np.array([0.4, 0.6], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 4: Batch of samples (2D x, 2D n of shape (2, 1), 2D p)
    x = np.array([[1, 2, 1], [0, 2, 2]], dtype=np.int32)
    n = np.array([[4], [4]], dtype=np.int32)
    p = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 5: K=5 categories, 2D x, 2D n of shape (3, 1), 1D p
    x = np.array([[2, 1, 0, 1, 1], [0, 0, 5, 0, 0], [1, 1, 1, 1, 1]], dtype=np.int32)
    n = np.array([[5], [5], [5]], dtype=np.int32)
    p = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 6: 1D int64 counts and float32 probability, scalar n
    x = np.array([1, 0, 0, 0], dtype=np.int64)
    n = np.array(1, dtype=np.int64)
    p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 7: High count integer array, scalar n
    x = np.array([10, 20, 30], dtype=np.int32)
    n = np.array(60, dtype=np.int32)
    p = np.array([0.1, 0.3, 0.6], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 8: 3D input arrays, 3D n of shape (2, 2, 1)
    x = np.array([[[1, 1, 1], [2, 1, 0]], [[0, 3, 0], [1, 1, 1]]], dtype=np.int32)
    n = np.array([[[3], [3]], [[3], [3]]], dtype=np.int32)
    p = np.array([[[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]], [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 9: 2D x, 2D n of shape (3, 1), 1D p
    x = np.array([[2, 2], [1, 3], [4, 0]], dtype=np.int32)
    n = np.array([[4], [4], [4]], dtype=np.int32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 10: 10 categories, single row, 2D n of shape (1, 1)
    x = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 0, 0]], dtype=np.int32)
    n = np.array([[10]], dtype=np.int32)
    p = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    return list_of_inputs

generated_inputs["jax.scipy.stats.multinomial.pmf_2"] = jax_scipy_stats_multinomial_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.multinomial.pmf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.multinomial.pmf_2'.")


check_valid('jax.scipy.stats.multinomial.pmf', generated_inputs['jax.scipy.stats.multinomial.pmf_2'], lib="jax", suffix=2)
