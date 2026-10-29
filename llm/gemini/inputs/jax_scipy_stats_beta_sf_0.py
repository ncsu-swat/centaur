
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def beta_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of standard beta
    input_dict = {
        "x": np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32),
        "a": np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0, 3.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with float64
    input_dict = {
        "x": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64),
        "a": np.array([[1.5, 2.5], [0.5, 3.0]], dtype=np.float64),
        "b": np.array([[2.5, 1.5], [3.0, 0.5]], dtype=np.float64),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting across dimensions
    input_dict = {
        "x": np.linspace(0.1, 0.9, 6, dtype=np.float32).reshape(2, 3),
        "a": np.array([[1.0], [2.0]], dtype=np.float32),
        "b": np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Non-zero loc and non-unit scale
    input_dict = {
        "x": np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Values of x below loc (should evaluate to 1.0)
    input_dict = {
        "x": np.array([-1.0, -0.5, 0.0], dtype=np.float32),
        "a": np.array([2.0, 3.0, 1.5], dtype=np.float32),
        "b": np.array([2.0, 1.0, 0.5], dtype=np.float32),
        "loc": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Values of x above loc + scale (should evaluate to 0.0)
    input_dict = {
        "x": np.array([3.5, 4.0, 5.0], dtype=np.float32),
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "scale": np.array([2.0, 2.0, 2.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative loc and positive scale
    input_dict = {
        "x": np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64),
        "a": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
        "b": np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64),
        "loc": np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float64),
        "scale": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D tensor inputs
    input_dict = {
        "x": np.full((2, 2, 2), 0.4, dtype=np.float32),
        "a": np.full((2, 2, 2), 3.0, dtype=np.float32),
        "b": np.full((2, 2, 2), 4.0, dtype=np.float32),
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large parameters a and b
    input_dict = {
        "x": np.array([50.5, 51.0, 52.0], dtype=np.float64),
        "a": np.array([100.0, 150.0, 200.0], dtype=np.float64),
        "b": np.array([100.0, 150.0, 200.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([100.0, 100.0, 100.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.beta.sf"] = beta_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.sf'.")


check_valid('jax.scipy.stats.beta.sf', generated_inputs['jax.scipy.stats.beta.sf'], lib="jax", suffix=0)
