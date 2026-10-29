
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hyp2f1_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    input_dict = {
        "a": np.array(0.5, dtype=np.float32),
        "b": np.array(0.5, dtype=np.float32),
        "c": np.array(1.5, dtype=np.float32),
        "x": np.array(0.2, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays (float32)
    input_dict = {
        "a": np.array([1.0, 1.5, 2.0], dtype=np.float32),
        "b": np.array([0.5, 1.0, 0.5], dtype=np.float32),
        "c": np.array([2.0, 2.5, 3.0], dtype=np.float32),
        "x": np.array([0.1, 0.3, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays (float32)
    input_dict = {
        "a": np.full((2, 3), 1.2, dtype=np.float32),
        "b": np.full((2, 3), 0.8, dtype=np.float32),
        "c": np.full((2, 3), 2.5, dtype=np.float32),
        "x": np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays with float64 precision
    input_dict = {
        "a": np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float64),
        "b": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        "c": np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float64),
        "x": np.array([0.05, 0.15, 0.25, 0.35], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays (float32)
    input_dict = {
        "a": np.ones((2, 2, 2), dtype=np.float32) * 0.5,
        "b": np.ones((2, 2, 2), dtype=np.float32) * 1.5,
        "c": np.ones((2, 2, 2), dtype=np.float32) * 3.0,
        "x": np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape(2, 2, 2),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes (3, 1) and (1, 4)
    input_dict = {
        "a": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        "b": np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32),
        "c": np.array([[4.0]], dtype=np.float32),
        "x": np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting scalar 0D array with 1D arrays
    input_dict = {
        "a": np.array(1.0, dtype=np.float64),
        "b": np.array(2.0, dtype=np.float64),
        "c": np.array(3.0, dtype=np.float64),
        "x": np.array([0.1, 0.2, 0.4, 0.6, 0.8], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small values near zero for x
    input_dict = {
        "a": np.array([0.1, 0.2], dtype=np.float32),
        "b": np.array([0.3, 0.4], dtype=np.float32),
        "c": np.array([1.0, 1.1], dtype=np.float32),
        "x": np.array([1e-4, 1e-3], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D arrays (float32)
    input_dict = {
        "a": np.full((1, 2, 1, 3), 0.5, dtype=np.float32),
        "b": np.full((1, 2, 1, 3), 0.5, dtype=np.float32),
        "c": np.full((1, 2, 1, 3), 2.0, dtype=np.float32),
        "x": np.full((1, 2, 1, 3), 0.25, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays with float64 precision
    input_dict = {
        "a": np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64),
        "b": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
        "c": np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float64),
        "x": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.special.hyp2f1"] = hyp2f1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.hyp2f1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.hyp2f1'.")


check_valid('jax.scipy.special.hyp2f1', generated_inputs['jax.scipy.special.hyp2f1'], lib="jax", suffix=0)
