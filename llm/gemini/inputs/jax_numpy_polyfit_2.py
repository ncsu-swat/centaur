
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyfit_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D fit, degree 1
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 1,
        "rcond": 1e-5,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D y (shape (M, K)), degree 2, float64
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array(
        [
            [4.0, 8.0],
            [1.0, 2.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [4.0, 8.0],
            [9.0, 18.0],
        ],
        dtype=np.float64,
    )
    w = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 2,
        "rcond": 1e-8,
        "full": True,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative and float values, degree 3
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float32)
    y = (x**3 - 2 * x**2 + x - 5).astype(np.float32)
    w = np.ones(10, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 3,
        "rcond": 1e-4,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Degree 0 (constant fit)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.5, 2.4, 2.6, 2.5, 2.5], dtype=np.float32)
    w = np.array([0.5, 1.0, 1.5, 1.0, 0.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 0,
        "rcond": 1e-6,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D y with multiple columns, full=True
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.stack([x * 2.0, x**2, -x + 3.0], axis=-1).astype(np.float32)
    w = np.ones(6, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 2,
        "rcond": 1e-5,
        "full": True,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher degree polynomial fit
    x = np.linspace(0.0, 2.0, 8, dtype=np.float64)
    y = np.exp(x).astype(np.float64)
    w = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.3, 0.7], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 4,
        "rcond": 1e-12,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 2D y, full=False
    x = np.array([0.1, 0.5, 1.2, 1.8, 2.3, 3.1, 4.0], dtype=np.float64)
    y = np.array(
        [
            [1.0, -1.0],
            [1.5, -0.5],
            [2.2, 0.2],
            [3.0, 1.0],
            [3.8, 1.8],
            [4.5, 2.5],
            [5.5, 3.5],
        ],
        dtype=np.float64,
    )
    w = np.ones(7, dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 1,
        "rcond": 1e-7,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Non-uniform weights
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    y = (x**2 + 3.0).astype(np.float32)
    w = np.array([0.1, 0.5, 1.0, 2.0, 1.0, 0.5, 0.1], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 2,
        "rcond": 1e-4,
        "full": True,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Degree 5 polynomial fit
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float64)
    y = (np.sin(x) + np.cos(x)).astype(np.float64)
    w = np.ones(12, dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 5,
        "rcond": 1e-10,
        "full": False,
        "w": w,
        "cov": "unscaled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D y single column
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    y = np.array([[100.0], [400.0], [900.0], [1600.0], [2500.0]], dtype=np.float32)
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "deg": 2,
        "rcond": 1e-3,
        "full": False,
        "w": w,
        "cov": "unscaled",
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
