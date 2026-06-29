
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def betabinom_pmf_inputs():
    list_of_inputs = []

    # Input 1: Standard 1D float32 arrays
    k = np.array([0, 1, 2, 3], dtype=np.float32)
    n = np.array([3, 3, 3, 3], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 2: Scalar arrays (0D)
    k = np.array(2, dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    a = np.array(1.5, dtype=np.float32)
    b = np.array(2.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 3: 2D arrays with float64
    k = np.array([[1, 2], [0, 3]], dtype=np.float64)
    n = np.array([[4, 4], [4, 4]], dtype=np.float64)
    a = np.array([[2.0, 3.0], [1.5, 2.5]], dtype=np.float64)
    b = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    loc = np.array([[0, 0], [0, 0]], dtype=np.float64)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 4: Broadcasting shapes
    k = np.array([1, 2, 3], dtype=np.int32)
    n = np.array([[5], [10]], dtype=np.int32)
    a = np.array([1.5], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 5: Non-zero loc parameter
    k = np.array([2, 3, 4], dtype=np.float32)
    n = np.array([5, 5, 5], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([1, 1, 1], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 6: Multi-dimensional, mixed types (integer and float)
    k = np.ones((2, 3, 2), dtype=np.int32)
    n = np.ones((2, 3, 2), dtype=np.int32) * 5
    a = np.full((2, 3, 2), 0.5, dtype=np.float32)
    b = np.full((2, 3, 2), 1.5, dtype=np.float32)
    loc = np.zeros((2, 3, 2), dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 7: Values of k that exceed limits (should still be valid to evaluate, yielding 0)
    k = np.array([-1, 0, 1, 10], dtype=np.float32)
    n = np.array([5, 5, 5, 5], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 8: High dimensions with random input satisfying constraints
    k = np.random.randint(0, 5, size=(2, 1, 3)).astype(np.float32)
    n = np.array([[[5, 6, 7]]], dtype=np.float32)
    a = np.array([2.5], dtype=np.float32)
    b = np.array([3.5], dtype=np.float32)
    loc = np.array([0], dtype=np.float32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 9: Small positive parameters (a, b near zero)
    k = np.array([1], dtype=np.int32)
    n = np.array([2], dtype=np.int32)
    a = np.array([0.01], dtype=np.float32)
    b = np.array([0.01], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    # Input 10: Larger numbers
    k = np.array([50], dtype=np.int32)
    n = np.array([100], dtype=np.int32)
    a = np.array([10.0], dtype=np.float64)
    b = np.array([20.0], dtype=np.float64)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append({"k": k, "n": n, "a": a, "b": b, "loc": loc})

    return list_of_inputs

generated_inputs["jax.scipy.stats.betabinom.pmf"] = betabinom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.betabinom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.betabinom.pmf'.")


check_valid('jax.scipy.stats.betabinom.pmf', generated_inputs['jax.scipy.stats.betabinom.pmf'], lib="jax", suffix=0)
