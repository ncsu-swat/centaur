
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multinomial_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D case with 3 categories
    x = np.array([1, 2, 3], dtype=np.int32)
    n = 6
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 2: Binary category with float64
    x = np.array([2, 0], dtype=np.int64)
    n = 2
    p = np.array([0.7, 0.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 3: 2D array of counts with shared 1D probabilities
    x = np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int32)
    n = 2
    p = np.array([0.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 4: 2D array of counts and matching 2D probabilities
    x = np.array([[3, 1, 0], [1, 2, 1]], dtype=np.int32)
    n = 4
    p = np.array(
        [[0.5, 0.25, 0.25], [0.1, 0.4, 0.5]], dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 5: 3D tensor input
    x = np.ones((2, 2, 3), dtype=np.int32)
    n = 3
    p = np.full((2, 2, 3), 1.0 / 3.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 6: Single-category outcome (boundary dimension)
    x = np.array([10], dtype=np.int32)
    n = 10
    p = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 7: Zero trials boundary case
    x = np.array([0, 0, 0], dtype=np.int32)
    n = 0
    p = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 8: 4 categories with int64 and float64
    x = np.array([2, 3, 1, 4], dtype=np.int64)
    n = 10
    p = np.array([0.1, 0.4, 0.2, 0.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 9: Extreme outcome where all counts are concentrated in one category
    x = np.array([0, 0, 0, 5], dtype=np.int32)
    n = 5
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 10: Broadcasting 1D count across 2D probability matrix
    x = np.array([1, 2, 1], dtype=np.int32)
    n = 4
    p = np.array(
        [[0.25, 0.5, 0.25], [0.1, 0.8, 0.1], [0.33, 0.34, 0.33]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.multinomial.pmf_1"] = multinomial_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.multinomial.pmf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.multinomial.pmf_1'.")


check_valid('jax.scipy.stats.multinomial.pmf', generated_inputs['jax.scipy.stats.multinomial.pmf_1'], lib="jax", suffix=1)
