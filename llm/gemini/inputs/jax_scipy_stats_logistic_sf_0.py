
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logistic_sf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars as arrays), float64
    input_dict = {
        "x": np.array(1.5, dtype=np.float64),
        "loc": np.array(-0.5, dtype=np.float64),
        "scale": np.array(2.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32, scale strictly positive
    input_dict = {
        "x": np.random.randn(2, 3).astype(np.float32),
        "loc": np.random.randn(2, 3).astype(np.float32),
        "scale": np.abs(np.random.randn(2, 3)).astype(np.float32) + 0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float64
    input_dict = {
        "x": np.random.randn(2, 2, 2).astype(np.float64),
        "loc": np.random.randn(2, 2, 2).astype(np.float64),
        "scale": np.abs(np.random.randn(2, 2, 2)).astype(np.float64) + 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcastable dimensions (x: 2D, loc: 1D, scale: scalar-like 1D)
    input_dict = {
        "x": np.random.randn(3, 4).astype(np.float32),
        "loc": np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        "scale": np.array([0.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values of x
    input_dict = {
        "x": np.array([100.0, -100.0, 50.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small positive scale values
    input_dict = {
        "x": np.array([0.5, -0.5], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.01, 0.05], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, float32
    input_dict = {
        "x": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "loc": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "scale": np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float32) + 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large scales
    input_dict = {
        "x": np.array([-10.0, 10.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([100.0, 1000.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative locations, float32
    input_dict = {
        "x": np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float32),
        "loc": np.array([[5.0, -5.0], [0.0, 1.0]], dtype=np.float32),
        "scale": np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.sf"] = logistic_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.sf'.")


check_valid('jax.scipy.stats.logistic.sf', generated_inputs['jax.scipy.stats.logistic.sf'], lib="jax", suffix=0)
