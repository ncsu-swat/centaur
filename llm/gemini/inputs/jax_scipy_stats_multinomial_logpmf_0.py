
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def multinomial_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with int32 and float32, scalar n
    x = np.array([1, 2, 3], dtype=np.int32)
    n = np.array(6, dtype=np.int32)
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 2: 1D arrays with int64 and float64, scalar n
    x = np.array([2, 2, 2], dtype=np.int64)
    n = np.array(6, dtype=np.int64)
    p = np.array([1/3, 1/3, 1/3], dtype=np.float64)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 3: Batch size 1, 2D arrays (int32 and float32), n shape (1, 1)
    x = np.array([[2, 3, 5]], dtype=np.int32)
    n = np.array([[10]], dtype=np.int32)
    p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 4: Batch size 2, 2D arrays (int64 and float32), n shape (2, 1)
    x = np.array([[1, 2], [3, 0]], dtype=np.int64)
    n = np.array([[3], [3]], dtype=np.int64)
    p = np.array([[0.4, 0.6], [0.5, 0.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 5: 3D arrays (batch of matrices), n shape (2, 2, 1)
    x = np.array([[[1, 1, 1], [2, 0, 1]], [[0, 3, 0], [1, 2, 0]]], dtype=np.int32)
    n = np.array([[[3], [3]], [[3], [3]]], dtype=np.int32)
    p = np.array([[[0.3, 0.3, 0.4], [0.2, 0.2, 0.6]], [[0.1, 0.8, 0.1], [0.5, 0.4, 0.1]]], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 6: Large integer values
    x = np.array([10, 20, 30, 40], dtype=np.int32)
    n = np.array(100, dtype=np.int32)
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 7: Single trial with float64 probability
    x = np.array([1, 0, 0], dtype=np.int32)
    n = np.array(1, dtype=np.int32)
    p = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 8: int64 for count x and trials n
    x = np.array([4, 0, 1], dtype=np.int64)
    n = np.array(5, dtype=np.int64)
    p = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 9: Single category (degenerate case)
    x = np.array([5], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    # Input 10: int64 and float64 batch, n shape (2, 1)
    x = np.array([[1, 1], [2, 2]], dtype=np.int64)
    n = np.array([[2], [4]], dtype=np.int64)
    p = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "n": n, "p": p})

    return list_of_inputs

generated_inputs["jax.scipy.stats.multinomial.logpmf"] = multinomial_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.multinomial.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.multinomial.logpmf'.")


check_valid('jax.scipy.stats.multinomial.logpmf', generated_inputs['jax.scipy.stats.multinomial.logpmf'], lib="jax", suffix=0)
