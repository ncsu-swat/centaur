
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def betabinom_pmf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar-like arrays
    input_dict = {
        "k": np.array(2, dtype=np.int32),
        "n": np.array(5, dtype=np.int32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1-D arrays with float32
    input_dict = {
        "k": np.array([0, 1, 2, 3], dtype=np.float32),
        "n": np.array([3, 3, 3, 3], dtype=np.float32),
        "a": np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2-D arrays with broadcasting
    input_dict = {
        "k": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        "n": np.array([[5], [10]], dtype=np.int32),
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([[2.0]], dtype=np.float32),
        "loc": np.array([[0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 precision with non-zero location offset
    input_dict = {
        "k": np.array([5.0, 6.0, 7.0], dtype=np.float64),
        "n": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "a": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "b": np.array([0.5, 1.0, 2.0], dtype=np.float64),
        "loc": np.array([5.0, 5.0, 5.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3-D arrays
    input_dict = {
        "k": np.ones((2, 3, 4), dtype=np.int32) * 2,
        "n": np.ones((2, 3, 4), dtype=np.int32) * 5,
        "a": np.full((2, 3, 4), 1.2, dtype=np.float32),
        "b": np.full((2, 3, 4), 3.4, dtype=np.float32),
        "loc": np.zeros((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative location parameter
    input_dict = {
        "k": np.array([-1, 0, 1], dtype=np.int32),
        "n": np.array([4, 4, 4], dtype=np.int32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "loc": np.array([-2, -2, -2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Values outside support (k < loc or k > n + loc)
    input_dict = {
        "k": np.array([10.0, -5.0, 20.0], dtype=np.float32),
        "n": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "a": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "b": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boundary case n = 0
    input_dict = {
        "k": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "n": np.array([[0, 0], [0, 0]], dtype=np.int32),
        "a": np.array([[2.5, 1.5], [0.5, 3.5]], dtype=np.float64),
        "b": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
        "loc": np.array([[0, 0], [0, 0]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2-D k with 1-D size-1 broadcasting parameters
    input_dict = {
        "k": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "n": np.array([5], dtype=np.float32),
        "a": np.array([3.0], dtype=np.float32),
        "b": np.array([4.0], dtype=np.float32),
        "loc": np.array([1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large shape parameters a and b with int64 / float64
    input_dict = {
        "k": np.array([5, 10, 15], dtype=np.int64),
        "n": np.array([20, 20, 20], dtype=np.int64),
        "a": np.array([100.0, 200.0, 300.0], dtype=np.float64),
        "b": np.array([100.0, 200.0, 300.0], dtype=np.float64),
        "loc": np.array([0, 0, 0], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.betabinom.pmf"] = betabinom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.betabinom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.betabinom.pmf'.")


check_valid('jax.scipy.stats.betabinom.pmf', generated_inputs['jax.scipy.stats.betabinom.pmf'], lib="jax", suffix=0)
