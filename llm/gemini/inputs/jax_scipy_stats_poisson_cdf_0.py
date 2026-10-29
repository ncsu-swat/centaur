
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def poisson_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, integer values
    k = np.array(3, dtype=np.int32)
    mu = np.array(2.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 2: 1D arrays, float32
    k = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    mu = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 3: Negative k values where CDF is 0
    k = np.array([-3, -1, 0, 2], dtype=np.int64)
    mu = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    loc = np.array([0, 0, 0, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 4: Non-zero loc parameter
    k = np.array([5, 10, 15], dtype=np.int32)
    mu = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    loc = np.array([2, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 5: 2D arrays
    k = np.array([[1, 2], [3, 4]], dtype=np.int32)
    mu = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 6: Broadcasting between shapes (3, 1) and (1, 4)
    k = np.array([[1], [3], [5]], dtype=np.float64)
    mu = np.array([[0.5, 1.0, 2.0, 4.0]], dtype=np.float64)
    loc = np.array([[0.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 7: Float64 precision with negative loc
    k = np.array([0.0, 1.5, 2.7, 4.2], dtype=np.float64)
    mu = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    loc = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 8: 3D arrays
    k = np.ones((2, 3, 4), dtype=np.int32) * 5
    mu = np.full((2, 3, 4), 3.0, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 9: Large mu values
    k = np.array([50, 100, 150], dtype=np.int64)
    mu = np.array([45.0, 100.0, 160.0], dtype=np.float64)
    loc = np.array([0, 0, 0], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 10: Non-integer k values (tested with fractional floats)
    k = np.array([0.2, 1.8, 2.5, 3.1], dtype=np.float32)
    mu = np.array([1.2, 1.2, 1.2, 1.2], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    # Input 11: 4D broadcasted inputs
    k = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 4)
    mu = np.array([[[[2.5]]]], dtype=np.float32)
    loc = np.array([[[[1.0]]]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "mu": mu, "loc": loc}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.poisson.cdf"] = poisson_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.poisson.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.poisson.cdf'.")


check_valid('jax.scipy.stats.poisson.cdf', generated_inputs['jax.scipy.stats.poisson.cdf'], lib="jax", suffix=0)
