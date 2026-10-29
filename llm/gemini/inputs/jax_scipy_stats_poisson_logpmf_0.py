
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def poisson_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalars (integers / floats)
    k = np.array(3, dtype=np.int32)
    mu = np.array(2.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 2: 1-D arrays
    k = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    loc = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 3: Non-zero loc parameter
    k = np.array([5, 6, 7], dtype=np.float32)
    mu = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 4: Negative k relative to loc (returns -inf)
    k = np.array([-1, 0, 1], dtype=np.int32)
    mu = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 5: 2-D matrices
    k = np.array([[0, 1], [2, 3]], dtype=np.int32)
    mu = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 6: Broadcasting shapes (3, 1) and (1, 4)
    k = np.array([[0], [1], [2]], dtype=np.float32)
    mu = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 7: Float64 precision
    k = np.array([10, 20, 30], dtype=np.float64)
    mu = np.array([15.5, 20.0, 25.5], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 8: 3-D tensors
    k = np.ones((2, 3, 4), dtype=np.int32) * 5
    mu = np.full((2, 3, 4), 4.2, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 9: mu=0 edge case
    k = np.array([0, 1, 2], dtype=np.int32)
    mu = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 10: Non-integer k values
    k = np.array([1.5, 2.7, 3.2], dtype=np.float32)
    mu = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    # Input 11: Negative loc values
    k = np.array([1, 2, 3], dtype=np.int32)
    mu = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    loc = np.array([-2, -2, -2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"k": k, "mu": mu, "loc": loc})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.poisson.logpmf"] = poisson_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.poisson.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.poisson.logpmf'.")


check_valid('jax.scipy.stats.poisson.logpmf', generated_inputs['jax.scipy.stats.poisson.logpmf'], lib="jax", suffix=0)
