
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_bernoulli_pmf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays
    k = np.array([0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 2: 0-D arrays (scalars as tensors)
    k = np.array(1, dtype=np.int32)
    p = np.array(0.3, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 3: Float values for k (some outside {0, 1})
    k = np.array([-0.5, 0.0, 1.0, 1.5], dtype=np.float32)
    p = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 4: 2D arrays
    k = np.array([[0, 1], [1, 0]], dtype=np.int32)
    p = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    loc = np.array([[0, 0], [0, 0]], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 5: Broadcasting - k 1D, p 2D, loc 1D
    k = np.array([0, 1], dtype=np.int32)
    p = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    loc = np.array([0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 6: Float64 precision
    k = np.array([1, 0, 1], dtype=np.float64)
    p = np.array([0.7, 0.7, 0.7], dtype=np.float64)
    loc = np.array([0, 0, 0], dtype=np.float64)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 7: 3D arrays
    k = np.zeros((2, 2, 2), dtype=np.int32)
    p = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    loc = np.zeros((2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 8: Non-zero location offset
    k = np.array([1, 2, 3], dtype=np.int32)
    p = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    loc = np.array([1, 1, 1], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 9: Edge cases of probability parameter (0.0 and 1.0)
    k = np.array([0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 10: Array of differing location parameters
    k = np.array([2, 2, 2], dtype=np.int32)
    p = np.array([0.3, 0.3, 0.3], dtype=np.float32)
    loc = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.scipy.stats.bernoulli.pmf"] = jax_scipy_stats_bernoulli_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.pmf'.")


check_valid('jax.scipy.stats.bernoulli.pmf', generated_inputs['jax.scipy.stats.bernoulli.pmf'], lib="jax", suffix=0)
