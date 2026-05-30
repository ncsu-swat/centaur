
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyfit_inputs():
    list_of_inputs = []

    # Input 1: Simple linear fit, 1D float32 arrays
    x1 = np.array([0., 1., 2., 3., 4.], dtype=np.float32)
    y1 = np.array([1., 3., 2., 5., 4.], dtype=np.float32)
    w1 = np.ones((5,), dtype=np.float32)
    list_of_inputs.append({
        'x': x1,
        'y': y1,
        'deg': 1,
        'rcond': 1e-5,
        'full': False,
        'w': w1,
        'cov': False
    })

    # Input 2: Quadratic fit with negative values, full output
    x2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y2 = np.array([4.1, 1.1, 0.2, 0.9, 3.9, 9.1], dtype=np.float32)
    w2 = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append({
        'x': x2,
        'y': y2,
        'deg': 2,
        'rcond': 1e-6,
        'full': True,
        'w': w2,
        'cov': False
    })

    # Input 3: Cubic fit, float64, returning covariance matrix
    x3 = np.linspace(-3.0, 3.0, 10, dtype=np.float64)
    y3 = np.sin(x3)
    w3 = np.ones((10,), dtype=np.float64)
    list_of_inputs.append({
        'x': x3,
        'y': y3,
        'deg': 3,
        'rcond': 1e-12,
        'full': False,
        'w': w3,
        'cov': True
    })

    # Input 4: Constant fit (deg=0)
    x4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y4 = np.array([2.5, 2.4, 2.6, 2.5], dtype=np.float32)
    w4 = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({
        'x': x4,
        'y': y4,
        'deg': 0,
        'rcond': 1e-5,
        'full': False,
        'w': w4,
        'cov': True
    })

    # Input 5: Linear fit with 2D y array (multiple columns)
    x5 = np.linspace(0, 10, 20, dtype=np.float32)
    y5 = np.zeros((20, 2), dtype=np.float32)
    y5[:, 0] = x5 * 2.0 + 1.0
    y5[:, 1] = -x5 * 0.5 + 3.0
    w5 = np.ones((20,), dtype=np.float32)
    list_of_inputs.append({
        'x': x5,
        'y': y5,
        'deg': 1,
        'rcond': 1e-7,
        'full': False,
        'w': w5,
        'cov': False
    })

    # Input 6: Higher-degree polynomial fit (deg=4), float64, full output
    x6 = np.linspace(-1, 1, 8, dtype=np.float64)
    y6 = x6**4 - 2 * x6**2
    w6 = np.linspace(0.1, 1.0, 8, dtype=np.float64)
    list_of_inputs.append({
        'x': x6,
        'y': y6,
        'deg': 4,
        'rcond': 1e-10,
        'full': True,
        'w': w6,
        'cov': False
    })

    # Input 7: Quadratic fit, returning covariance
    x7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y7 = np.array([1.5, 1.7, 2.0, 2.5, 3.2], dtype=np.float32)
    w7 = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        'x': x7,
        'y': y7,
        'deg': 2,
        'rcond': 1e-5,
        'full': False,
        'w': w7,
        'cov': True
    })

    # Input 8: Quadratic fit, 2D float64 y array, full=True
    x8 = np.linspace(0, 5, 12, dtype=np.float64)
    y8 = np.zeros((12, 3), dtype=np.float64)
    y8[:, 0] = x8**2 - 2
    y8[:, 1] = 2 * x8 + 1
    y8[:, 2] = -x8**2 + 5
    w8 = np.ones((12,), dtype=np.float64)
    list_of_inputs.append({
        'x': x8,
        'y': y8,
        'deg': 2,
        'rcond': 1e-9,
        'full': True,
        'w': w8,
        'cov': True
    })

    # Input 9: Quadratic fit with float32 and custom weights
    x9 = np.array([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    y9 = np.array([0.5, -0.5, -0.5, 0.5, 2.5, 5.5], dtype=np.float32)
    w9 = np.array([1.0, 1.2, 0.8, 1.0, 1.1, 0.9], dtype=np.float32)
    list_of_inputs.append({
        'x': x9,
        'y': y9,
        'deg': 2,
        'rcond': 1e-6,
        'full': False,
        'w': w9,
        'cov': False
    })

    # Input 10: Cubic fit with larger sample size, float32, with covariance
    x10 = np.linspace(-2.0, 2.0, 15, dtype=np.float32)
    y10 = x10**3 + x10**2 - x10
    w10 = np.ones((15,), dtype=np.float32)
    list_of_inputs.append({
        'x': x10,
        'y': y10,
        'deg': 3,
        'rcond': 1e-5,
        'full': False,
        'w': w10,
        'cov': True
    })

    return list_of_inputs

generated_inputs["jax.numpy.polyfit_1"] = polyfit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyfit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyfit_1'.")


check_valid('jax.numpy.polyfit', generated_inputs['jax.numpy.polyfit_1'], lib="jax", suffix=1)
