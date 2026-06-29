
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gamma_cdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "loc": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "scale": np.array([1.0, 1.5, 2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar-like 0D float64 arrays
    input_dict = {
        "x": np.array(2.5, dtype=np.float64),
        "a": np.array(3.0, dtype=np.float64),
        "loc": np.array(0.0, dtype=np.float64),
        "scale": np.array(1.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with random values
    input_dict = {
        "x": np.random.uniform(0.1, 5.0, size=(2, 3)).astype(np.float32),
        "a": np.random.uniform(0.5, 3.0, size=(2, 3)).astype(np.float32),
        "loc": np.zeros((2, 3), dtype=np.float32),
        "scale": np.ones((2, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative x values to test boundary conditions
    input_dict = {
        "x": np.array([-1.0, -0.5, 0.0], dtype=np.float32),
        "a": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting compatibility test
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "a": np.array([2.0], dtype=np.float32),
        "loc": np.array([[0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0], [2.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large positive x values (high probability region)
    input_dict = {
        "x": np.array([10.0, 100.0], dtype=np.float64),
        "a": np.array([2.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small shape parameters (a < 1)
    input_dict = {
        "x": np.array([0.1, 0.5, 1.0], dtype=np.float32),
        "a": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Location shifted distribution (x > loc)
    input_dict = {
        "x": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "loc": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 arrays
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.float32) * 3.0,
        "a": np.ones((2, 2, 2), dtype=np.float32) * 2.0,
        "loc": np.zeros((2, 2, 2), dtype=np.float32),
        "scale": np.ones((2, 2, 2), dtype=np.float32) * 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Diverse scale values
    input_dict = {
        "x": np.array([1.0, 1.0], dtype=np.float32),
        "a": np.array([3.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([0.1, 10.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gamma.cdf"] = gamma_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.cdf'.")


check_valid('jax.scipy.stats.gamma.cdf', generated_inputs['jax.scipy.stats.gamma.cdf'], lib="jax", suffix=0)
