
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multinomial_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with 3 categories, int32
    x = np.array([2, 1, 1], dtype=np.int32)
    n = np.array(4, dtype=np.int32)
    p = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 2: 1D with some zeros in x
    x = np.array([0, 0, 5], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 3: 2D batched x with column vector n (2, 1)
    x = np.array([[1, 2, 3], [2, 2, 2]], dtype=np.int32)
    n = np.array([[6], [6]], dtype=np.int32)
    p = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 4: 2D batched x with scalar n
    x = np.array([[3, 0], [1, 2]], dtype=np.int32)
    n = np.array(3, dtype=np.int32)
    p = np.array([0.4, 0.6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 5: int64 types for x and n, float64 for p
    x = np.array([4, 6], dtype=np.int64)
    n = np.array(10, dtype=np.int64)
    p = np.array([0.4, 0.6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 6: Single category (k=1)
    x = np.array([5], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 7: 3D batched tensor with broadcastable column n
    x = np.array([[[1, 1], [0, 2]], [[2, 0], [1, 1]]], dtype=np.int32)
    n = np.array([[[2], [2]], [[2], [2]]], dtype=np.int32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 8: Batched p matching 2D batched x
    x = np.array([[2, 1], [1, 2]], dtype=np.int32)
    n = np.array([[3], [3]], dtype=np.int32)
    p = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 9: High-dimensional category vector (k=10)
    x = np.ones(10, dtype=np.int32)
    n = np.array(10, dtype=np.int32)
    p = np.full(10, 0.1, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 10: Zero trials (n = 0)
    x = np.array([0, 0], dtype=np.int32)
    n = np.array(0, dtype=np.int32)
    p = np.array([0.3, 0.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 11: int16 types with 4 categories
    x = np.array([10, 20, 30, 40], dtype=np.int16)
    n = np.array(100, dtype=np.int16)
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.multinomial.pmf_2"] = multinomial_pmf_inputs()

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
