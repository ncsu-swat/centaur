
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multinomial_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with 3 categories (int32)
    x = np.array([2, 1, 1], dtype=np.int32)
    n = np.array(4, dtype=np.int32)
    p = np.array([0.25, 0.25, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 2: 1D with 2 categories (int64)
    x = np.array([1, 2], dtype=np.int64)
    n = np.array(3, dtype=np.int64)
    p = np.array([0.4, 0.6], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 3: 2D batch of observations with scalar n
    x = np.array([[1, 2, 0], [0, 1, 2]], dtype=np.int32)
    n = np.array(3, dtype=np.int32)
    p = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 4: 2D batch with corresponding 2D probabilities (float64)
    x = np.array([[2, 1], [1, 2]], dtype=np.int64)
    n = np.array(3, dtype=np.int64)
    p = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 5: Larger number of categories (k=5)
    x = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    n = np.array(10, dtype=np.int32)
    p = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 6: 3D tensor batch with scalar n
    x = np.array(
        [[[1, 1], [2, 0]], [[0, 2], [1, 1]]],
        dtype=np.int32,
    )
    n = np.array(2, dtype=np.int32)
    p = np.array([0.6, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 7: Observations with zeros in some categories
    x = np.array([0, 0, 5], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 8: Zero total trials (n=0)
    x = np.array([0, 0, 0], dtype=np.int32)
    n = np.array(0, dtype=np.int32)
    p = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 9: Uniform probabilities with float64 precision
    x = np.array([1, 2, 3], dtype=np.int64)
    n = np.array(6, dtype=np.int64)
    p = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 10: Single category (k=1)
    x = np.array([5], dtype=np.int32)
    n = np.array(5, dtype=np.int32)
    p = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    # Input 11: Larger counts with int64 and float64
    x = np.array([100, 200, 300], dtype=np.int64)
    n = np.array(600, dtype=np.int64)
    p = np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "n": n, "p": p}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.multinomial.logpmf"] = (
    multinomial_logpmf_inputs()
)

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
