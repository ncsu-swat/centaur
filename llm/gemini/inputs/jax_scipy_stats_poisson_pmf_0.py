
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_poisson_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, integers for k, float32 for mu and loc
    k = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    mu = np.array([2.5], dtype=np.float32)
    loc = np.array([0], dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32
    k = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mu = np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 1D arrays with non-zero loc
    k = np.array([10, 20, 30], dtype=np.float64)
    mu = np.array([15.0, 15.0, 15.0], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays
    k = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    mu = np.ones((2, 2, 2), dtype=np.float32) * 3.5
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted shapes, k is (3, 1), mu is (1, 4), loc is (1, 1)
    k = np.array([[0], [2], [5]], dtype=np.int32)
    mu = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero-dimensional arrays (scalars as tensors)
    k = np.array(3, dtype=np.int32)
    mu = np.array(4.2, dtype=np.float32)
    loc = np.array(-1, dtype=np.int32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fractional values for k
    k = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    mu = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small mu values
    k = np.array([0, 1], dtype=np.int32)
    mu = np.array([0.01, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large mu values
    k = np.array([100, 110], dtype=np.int32)
    mu = np.array([100.0, 100.0], dtype=np.float32)
    loc = np.array([10, 10], dtype=np.int32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed precision types
    k = np.array([5, 10], dtype=np.int64)
    mu = np.array([4.5, 8.2], dtype=np.float64)
    loc = np.array([0, 1], dtype=np.int32)
    input_dict = {"k": k, "mu": mu, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.poisson.pmf"] = jax_scipy_stats_poisson_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.poisson.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.poisson.pmf'.")


check_valid('jax.scipy.stats.poisson.pmf', generated_inputs['jax.scipy.stats.poisson.pmf'], lib="jax", suffix=0)
