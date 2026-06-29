
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def betabinom_logpmf_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case (1D float32)
    list_of_inputs.append({
        "k": np.array([2.0], dtype=np.float32),
        "n": np.array([5.0], dtype=np.float32),
        "a": np.array([2.0], dtype=np.float32),
        "b": np.array([3.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32)
    })

    # Input 2: 1D arrays with multiple elements, integer values for k, n, loc
    list_of_inputs.append({
        "k": np.array([0, 1, 2, 3], dtype=np.int32),
        "n": np.array([5, 5, 5, 5], dtype=np.int32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([0, 0, 0, 0], dtype=np.int32)
    })

    # Input 3: Float64 inputs
    list_of_inputs.append({
        "k": np.array([5.0, 10.0], dtype=np.float64),
        "n": np.array([10.0, 20.0], dtype=np.float64),
        "a": np.array([1.5, 2.5], dtype=np.float64),
        "b": np.array([3.5, 4.5], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64)
    })

    # Input 4: 2D array inputs
    list_of_inputs.append({
        "k": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "n": np.array([[5, 5], [5, 5]], dtype=np.int32),
        "a": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        "b": np.array([[5.0, 4.0], [3.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0, 0], [0, 0]], dtype=np.int32)
    })

    # Input 5: Non-zero loc parameter
    list_of_inputs.append({
        "k": np.array([5, 6, 7], dtype=np.int32),
        "n": np.array([10, 10, 10], dtype=np.int32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 3.0, 3.0], dtype=np.float32),
        "loc": np.array([3, 3, 3], dtype=np.int32)
    })

    # Input 6: Shape parameters < 1
    list_of_inputs.append({
        "k": np.array([1.0], dtype=np.float32),
        "n": np.array([5.0], dtype=np.float32),
        "a": np.array([0.5], dtype=np.float32),
        "b": np.array([0.5], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32)
    })

    # Input 7: Large shape parameters and large values
    list_of_inputs.append({
        "k": np.array([100], dtype=np.int32),
        "n": np.array([200], dtype=np.int32),
        "a": np.array([50.0], dtype=np.float32),
        "b": np.array([50.0], dtype=np.float32),
        "loc": np.array([0], dtype=np.int32)
    })

    # Input 8: 3D array inputs
    list_of_inputs.append({
        "k": np.ones((2, 2, 2), dtype=np.int32) * 2,
        "n": np.ones((2, 2, 2), dtype=np.int32) * 5,
        "a": np.ones((2, 2, 2), dtype=np.float32) * 1.5,
        "b": np.ones((2, 2, 2), dtype=np.float32) * 2.5,
        "loc": np.zeros((2, 2, 2), dtype=np.int32)
    })

    # Input 9: High variance (large b, small a)
    list_of_inputs.append({
        "k": np.array([0, 1], dtype=np.int32),
        "n": np.array([10, 10], dtype=np.int32),
        "a": np.array([0.1, 0.1], dtype=np.float32),
        "b": np.array([10.0, 10.0], dtype=np.float32),
        "loc": np.array([0, 0], dtype=np.int32)
    })

    # Input 10: Broadcasting compatible inputs (scalar equivalent arrays)
    list_of_inputs.append({
        "k": np.array([[3]], dtype=np.int32),
        "n": np.array([[10]], dtype=np.int32),
        "a": np.array([[2.5]], dtype=np.float32),
        "b": np.array([[1.5]], dtype=np.float32),
        "loc": np.array([[1]], dtype=np.int32)
    })

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
