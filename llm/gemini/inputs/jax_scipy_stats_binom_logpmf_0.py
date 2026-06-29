
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_binom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like values (0D arrays)
    k = np.array(2, dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays
    k = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    n = np.array([4, 4, 4, 4, 4], dtype=np.int32)
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 inputs for all parameters
    k = np.array([1.0, 2.0], dtype=np.float32)
    n = np.array([10.0, 10.0], dtype=np.float32)
    p = np.array([0.25, 0.75], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays with float64 probability
    k = np.array([[1, 2], [3, 4]], dtype=np.int32)
    n = np.array([[10, 10], [10, 10]], dtype=np.int32)
    p = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    loc = np.array([[0, 0], [0, 0]], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted shapes (k: 1D, n: 1D, p: 2D, loc: 1D)
    k = np.array([1, 2, 3], dtype=np.int32)
    n = np.array([10], dtype=np.int32)
    p = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative k values
    k = np.array([-1, -2, 1], dtype=np.int32)
    n = np.array([5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: k greater than n
    k = np.array([6, 7, 8], dtype=np.int32)
    n = np.array([5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Positive offsets (loc != 0)
    k = np.array([3, 4, 5], dtype=np.int32)
    n = np.array([5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D Arrays
    k = np.ones((2, 2, 2), dtype=np.int32)
    n = np.full((2, 2, 2), 5, dtype=np.int32)
    p = np.full((2, 2, 2), 0.3, dtype=np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Edge case probabilities (p=0.0 and p=1.0)
    k = np.array([0, 5], dtype=np.int32)
    n = np.array([5, 5], dtype=np.int32)
    p = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.binom.logpmf"] = jax_scipy_stats_binom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.binom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.binom.logpmf'.")


check_valid('jax.scipy.stats.binom.logpmf', generated_inputs['jax.scipy.stats.binom.logpmf'], lib="jax", suffix=0)
