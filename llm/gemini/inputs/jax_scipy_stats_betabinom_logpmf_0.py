
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def betabinom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar arrays, integer k and n, float a and b
    input_dict = {
        "k": np.array(3, dtype=np.int32),
        "n": np.array(10, dtype=np.int32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D arrays, standard float64 values
    input_dict = {
        "k": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        "n": np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
        "a": np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float64),
        "b": np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With non-zero positive loc
    input_dict = {
        "k": np.array([2, 3, 4, 5], dtype=np.int32),
        "n": np.array([5, 5, 5, 5], dtype=np.int32),
        "a": np.array([0.5, 1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([3.0, 2.0, 1.0, 0.5], dtype=np.float32),
        "loc": np.array([2, 2, 2, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With negative loc
    input_dict = {
        "k": np.array([-2, -1, 0, 1], dtype=np.int64),
        "n": np.array([4, 4, 4, 4], dtype=np.int64),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
        "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64),
        "loc": np.array([-2, -2, -2, -2], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2-D arrays
    input_dict = {
        "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "n": np.array([[5, 5], [5, 5]], dtype=np.int32),
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b": np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float32),
        "loc": np.array([[0, 0], [0, 0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasted dimensions (k: (3, 1), n: (1, 4), a: (1,), b: (3, 4), loc: (1, 1))
    input_dict = {
        "k": np.array([[0], [1], [2]], dtype=np.float32),
        "n": np.array([[4, 5, 6, 7]], dtype=np.float32),
        "a": np.array([2.5], dtype=np.float32),
        "b": np.ones((3, 4), dtype=np.float32) * 1.5,
        "loc": np.array([[0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3-D arrays
    input_dict = {
        "k": np.ones((2, 3, 4), dtype=np.int32) * 2,
        "n": np.ones((2, 3, 4), dtype=np.int32) * 6,
        "a": np.full((2, 3, 4), 3.0, dtype=np.float32),
        "b": np.full((2, 3, 4), 5.0, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boundary points where k - loc == 0 and k - loc == n
    input_dict = {
        "k": np.array([0, 8], dtype=np.int32),
        "n": np.array([8, 8], dtype=np.int32),
        "a": np.array([1.2, 3.4], dtype=np.float64),
        "b": np.array([4.3, 2.1], dtype=np.float64),
        "loc": np.array([0, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large parameter values
    input_dict = {
        "k": np.array([50, 100], dtype=np.int64),
        "n": np.array([100, 200], dtype=np.int64),
        "a": np.array([50.0, 100.0], dtype=np.float64),
        "b": np.array([50.0, 100.0], dtype=np.float64),
        "loc": np.array([0, 0], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed broadcasting shapes with varied loc per element
    input_dict = {
        "k": np.array([[1], [3]], dtype=np.float64),
        "n": np.array([6, 8], dtype=np.float64),
        "a": np.array([0.8, 1.2], dtype=np.float64),
        "b": np.array([2.5, 3.5], dtype=np.float64),
        "loc": np.array([[0, 1], [1, 2]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.betabinom.logpmf"] = betabinom_logpmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.betabinom.logpmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.betabinom.logpmf'.")


check_valid('jax.scipy.stats.betabinom.logpmf', generated_inputs['jax.scipy.stats.betabinom.logpmf'], lib="jax", suffix=0)
