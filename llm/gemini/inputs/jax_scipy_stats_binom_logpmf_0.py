
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def binom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar arrays, standard values
    k = np.array(2, dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 2: 1-D arrays of matching length
    k = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    n = np.array([5, 5, 5, 5, 5, 5], dtype=np.int32)
    p = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    loc = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 3: 2-D arrays, float32 precision
    k = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    p = np.array([[0.25, 0.25], [0.5, 0.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 4: Broadcasting shapes: k (3, 1), n (1, 4), p (1, 1), loc (1, 1)
    k = np.array([[1], [2], [3]], dtype=np.int32)
    n = np.array([[5, 6, 7, 8]], dtype=np.int32)
    p = np.array([[0.4]], dtype=np.float32)
    loc = np.array([[0]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 5: Non-zero positive location offset
    k = np.array([5, 6, 7], dtype=np.int64)
    n = np.array([10, 10, 10], dtype=np.int64)
    p = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    loc = np.array([4, 4, 4], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 6: Negative location offset
    k = np.array([0, 1, 2], dtype=np.int32)
    n = np.array([5, 5, 5], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([-2, -2, -2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 7: Boundary success probability p = 0.0
    k = np.array([0, 1], dtype=np.int32)
    n = np.array([5, 5], dtype=np.int32)
    p = np.array([0.0, 0.0], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 8: Boundary success probability p = 1.0
    k = np.array([5, 4], dtype=np.int32)
    n = np.array([5, 5], dtype=np.int32)
    p = np.array([1.0, 1.0], dtype=np.float32)
    loc = np.array([0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 9: 3-D tensor inputs
    k = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    n = np.ones((2, 3, 4), dtype=np.float32) * 10.0
    p = np.full((2, 3, 4), 0.3, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    # Input 10: Float64 high precision evaluation
    k = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    n = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "n": n, "p": p, "loc": loc})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.binom.logpmf"] = binom_logpmf_inputs()

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
