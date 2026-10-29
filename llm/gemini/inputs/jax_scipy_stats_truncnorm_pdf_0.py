
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def truncnorm_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(-1.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays (float64)
    input_dict = {
        "x": np.array([-1.5, 0.0, 1.5], dtype=np.float64),
        "a": np.array([-2.0, -2.0, -2.0], dtype=np.float64),
        "b": np.array([2.0, 2.0, 2.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcastable shapes (1D x, scalar parameters)
    input_dict = {
        "x": np.linspace(-3.0, 3.0, 10, dtype=np.float32),
        "a": np.array(-2.0, dtype=np.float32),
        "b": np.array(2.0, dtype=np.float32),
        "loc": np.array(0.5, dtype=np.float32),
        "scale": np.array(1.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays with varying bounds
    input_dict = {
        "x": np.zeros((3, 3), dtype=np.float32),
        "a": np.full((3, 3), -1.0, dtype=np.float32),
        "b": np.full((3, 3), 1.0, dtype=np.float32),
        "loc": np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
        "scale": np.ones((3, 3), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Points outside the truncation interval [a, b]
    input_dict = {
        "x": np.array([-5.0, -3.0, 0.0, 3.0, 5.0], dtype=np.float64),
        "a": np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64),
        "b": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array inputs
    shape = (2, 3, 4)
    input_dict = {
        "x": np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32),
        "a": np.full(shape, -2.0, dtype=np.float32),
        "b": np.full(shape, 2.0, dtype=np.float32),
        "loc": np.zeros(shape, dtype=np.float32),
        "scale": np.full(shape, 0.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One-sided truncation (strictly positive side)
    input_dict = {
        "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "a": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([np.inf, np.inf, np.inf], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 2.0, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One-sided truncation (strictly negative side)
    input_dict = {
        "x": np.array([-3.0, -2.0, -1.0], dtype=np.float64),
        "a": np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
        "b": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-zero location and scale with broadcasting
    input_dict = {
        "x": np.array([[10.0], [12.0], [15.0]], dtype=np.float32),
        "a": np.array([-3.0], dtype=np.float32),
        "b": np.array([3.0], dtype=np.float32),
        "loc": np.array([[10.0, 12.0]], dtype=np.float32),
        "scale": np.array([2.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Asymmetric truncation bounds
    input_dict = {
        "x": np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        "a": np.array([-0.5, 0.0, 1.0, -2.0], dtype=np.float32),
        "b": np.array([1.5, 2.0, 3.0, 0.0], dtype=np.float32),
        "loc": np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float32),
        "scale": np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.scipy.stats.truncnorm.pdf"] = truncnorm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.truncnorm.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.truncnorm.pdf'.")


check_valid('jax.scipy.stats.truncnorm.pdf', generated_inputs['jax.scipy.stats.truncnorm.pdf'], lib="jax", suffix=0)
