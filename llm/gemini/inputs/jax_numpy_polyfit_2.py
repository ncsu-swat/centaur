
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyfit_inputs():
    list_of_inputs = []
    
    # Input 1: Basic quadratic fit with single output
    x = np.linspace(-5.0, 5.0, 10).astype(np.float32)
    y = (2.0 * x**2 + 3.0 * x + 1.0 + np.random.randn(10) * 0.1).astype(np.float32)
    w = np.random.uniform(0.5, 1.5, 10).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 2,
        'rcond': 1e-7,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Cubic fit with 2D outputs, float64
    x = np.linspace(-10.0, 10.0, 15).astype(np.float64)
    y = np.stack([x**3, x**2, x], axis=-1).astype(np.float64)
    w = np.ones(15, dtype=np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 3,
        'rcond': 1e-5,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Linear fit with 'full' output
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dtype=np.float32)
    w = np.ones(8, dtype=np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 1,
        'rcond': 1e-6,
        'full': True,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large 1D array, 'full' output
    x = np.random.randn(100).astype(np.float64)
    y = np.random.randn(100, 2).astype(np.float64)
    w = np.random.uniform(0.1, 2.0, 100).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 1,
        'rcond': 1e-8,
        'full': True,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High degree polynomial, negative inputs
    x = np.random.uniform(-10.0, 10.0, 20).astype(np.float32)
    y = np.random.uniform(-5.0, 5.0, 20).astype(np.float32)
    w = np.random.uniform(0.1, 1.0, 20).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 4,
        'rcond': 1e-4,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multi-dimensional target with quadratic degree
    x = np.random.randn(50).astype(np.float32)
    y = np.random.randn(50, 5).astype(np.float32)
    w = np.random.uniform(0.2, 1.8, 50).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 2,
        'rcond': 1e-3,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Degree-0 constant fit, float64
    x = np.random.randn(12).astype(np.float64)
    y = np.random.randn(12).astype(np.float64)
    w = np.random.uniform(0.5, 1.5, 12).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 0,
        'rcond': 1e-12,
        'full': True,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High degree 5 polynomial
    x = np.random.randn(30).astype(np.float32)
    y = np.random.randn(30).astype(np.float32)
    w = np.random.uniform(0.1, 2.0, 30).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 5,
        'rcond': 1e-5,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Exact fit where degree is 3, data points is 4
    x = np.random.randn(4).astype(np.float32)
    y = np.random.randn(4, 2).astype(np.float32)
    w = np.random.uniform(0.5, 1.5, 4).astype(np.float32)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 3,
        'rcond': 1e-7,
        'full': False,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger scale fit, float64
    x = np.random.randn(200).astype(np.float64)
    y = np.random.randn(200, 10).astype(np.float64)
    w = np.random.uniform(0.1, 10.0, 200).astype(np.float64)
    input_dict = {
        'x': x,
        'y': y,
        'deg': 2,
        'rcond': 1e-9,
        'full': True,
        'w': w,
        'cov': 'unscaled'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polyfit_2"] = polyfit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyfit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyfit_2'.")


check_valid('jax.numpy.polyfit', generated_inputs['jax.numpy.polyfit_2'], lib="jax", suffix=2)
