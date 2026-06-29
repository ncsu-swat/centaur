
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_multinomial_pmf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D arrays with int32/float32
    input_dict = {
        "x": np.array([1, 2, 1], dtype=np.int32),
        "n": 4,
        "p": np.array([0.2, 0.5, 0.3], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4 categories, sum to 6
    input_dict = {
        "x": np.array([3, 0, 1, 2], dtype=np.int32),
        "n": 6,
        "p": np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D x array, broadcasting with 1D p array
    input_dict = {
        "x": np.array([[1, 2], [2, 1], [0, 3]], dtype=np.int32),
        "n": 3,
        "p": np.array([0.4, 0.6], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Higher precision float64 and int64
    input_dict = {
        "x": np.array([10, 20, 30], dtype=np.int64),
        "n": 60,
        "p": np.array([0.15, 0.35, 0.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single category boundary case
    input_dict = {
        "x": np.array([5], dtype=np.int32),
        "n": 5,
        "p": np.array([1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Equal probabilities
    input_dict = {
        "x": np.array([5, 5, 5, 5], dtype=np.int32),
        "n": 20,
        "p": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small 2D case
    input_dict = {
        "x": np.array([[2, 1], [1, 2]], dtype=np.int32),
        "n": 3,
        "p": np.array([0.7, 0.3], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero count in some categories
    input_dict = {
        "x": np.array([0, 0, 0, 5], dtype=np.int64),
        "n": 5,
        "p": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Identity-like 2D counts with sum=1
    input_dict = {
        "x": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32),
        "n": 1,
        "p": np.array([0.2, 0.5, 0.3], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger numbers with float64
    input_dict = {
        "x": np.array([4, 4, 2], dtype=np.int32),
        "n": 10,
        "p": np.array([0.4, 0.4, 0.2], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.multinomial.pmf_1"] = jax_scipy_stats_multinomial_pmf_inputs()

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
