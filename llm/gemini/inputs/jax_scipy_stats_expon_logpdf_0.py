
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 arrays with location shift and scaling
    input_dict = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "loc": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        "scale": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Testing lower bound boundary conditions (x < loc)
    input_dict = {
        "x": np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting (loc and scale are 1D, x is 2D)
    input_dict = {
        "x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "loc": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "scale": np.array([1.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays with randomized valid values
    input_dict = {
        "x": np.random.uniform(2.0, 10.0, size=(2, 2, 2)).astype(np.float32),
        "loc": np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32),
        "scale": np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar-like (0D) arrays
    input_dict = {
        "x": np.array(2.5, dtype=np.float32),
        "loc": np.array(1.0, dtype=np.float32),
        "scale": np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale values
    input_dict = {
        "x": np.array([10.0, 100.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([50.0, 200.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small scale values with float64 precision
    input_dict = {
        "x": np.array([0.1, 0.2], dtype=np.float64),
        "loc": np.array([0.0, 0.0], dtype=np.float64),
        "scale": np.array([0.01, 0.05], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional (4D) inputs
    input_dict = {
        "x": np.random.uniform(1.0, 2.0, size=(2, 2, 1, 3)).astype(np.float32),
        "loc": np.zeros((2, 2, 1, 3), dtype=np.float32),
        "scale": np.ones((2, 2, 1, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive coordinates with varying scales
    input_dict = {
        "x": np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float32),
        "loc": np.array([[-2.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[0.5, 1.5], [1.0, 2.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.logpdf"] = expon_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.logpdf'.")


check_valid('jax.scipy.stats.expon.logpdf', generated_inputs['jax.scipy.stats.expon.logpdf'], lib="jax", suffix=0)
