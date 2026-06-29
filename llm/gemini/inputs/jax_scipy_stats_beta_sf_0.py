
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def beta_sf_inputs():
    list_of_inputs = []

    # Input 1: Basic scalars (represented as 0-D arrays) of float32
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of float32
    input_dict = {
        "x": np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32),
        "a": np.array([1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float32),
        "b": np.array([3.5, 3.0, 2.5, 2.0, 1.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays of float64
    input_dict = {
        "x": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float64),
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "b": np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float64),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting shapes
    input_dict = {
        "x": np.array([[0.2], [0.5], [0.8]], dtype=np.float32),  # (3, 1)
        "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),   # (4,)
        "b": np.array([2.0], dtype=np.float32),                  # (1,)
        "loc": np.array([0.0], dtype=np.float32),                # (1,)
        "scale": np.array([1.0], dtype=np.float32)               # (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large parameters (float32)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(100.0, dtype=np.float32),
        "b": np.array(100.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Non-standard loc and scale
    input_dict = {
        "x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "scale": np.array([4.0, 4.0, 4.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x below loc (should return 1)
    input_dict = {
        "x": np.array([-0.5, -1.0], dtype=np.float32),
        "a": np.array([2.0, 2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x above loc + scale (should return 0)
    input_dict = {
        "x": np.array([1.5, 2.0], dtype=np.float32),
        "a": np.array([2.0, 2.0], dtype=np.float32),
        "b": np.array([2.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Higher dimensional arrays (3D)
    input_dict = {
        "x": np.random.uniform(0.1, 0.9, size=(2, 2, 2)).astype(np.float32),
        "a": np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float32),
        "b": np.random.uniform(0.5, 5.0, size=(2, 2, 2)).astype(np.float32),
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Shape parameters smaller than 1.0
    input_dict = {
        "x": np.array([0.25, 0.75], dtype=np.float32),
        "a": np.array([0.5, 0.1], dtype=np.float32),
        "b": np.array([0.5, 0.1], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
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
