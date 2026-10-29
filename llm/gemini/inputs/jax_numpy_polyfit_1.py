
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polyfit_inputs():
    list_of_inputs = []

    # Input 1: 1D y, deg=1, full=False, cov=False, float64
    M = 10
    x = np.linspace(-5.0, 5.0, M, dtype=np.float64)
    y = (2.5 * x + 1.0 + np.random.randn(M) * 0.1).astype(np.float64)
    w = np.ones(M, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 1,
            "rcond": 1e-6,
            "full": False,
            "w": w,
            "cov": False,
        }
    )

    # Input 2: 1D y, deg=2, full=True, cov=False, float32
    M = 8
    x = np.linspace(0.0, 4.0, M, dtype=np.float32)
    y = (x**2 - 3.0 * x + 2.0).astype(np.float32)
    w = np.array([0.5, 1.0, 1.2, 0.8, 1.0, 1.5, 0.9, 1.1], dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 2,
            "rcond": 1e-5,
            "full": True,
            "w": w,
            "cov": False,
        }
    )

    # Input 3: 2D y with shape (M, 2), deg=1, full=False, cov=False
    M = 12
    x = np.linspace(1.0, 10.0, M, dtype=np.float32)
    y = np.column_stack([2.0 * x + 0.5, -1.5 * x + 3.0]).astype(np.float32)
    w = np.ones(M, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 1,
            "rcond": 1e-5,
            "full": False,
            "w": w,
            "cov": False,
        }
    )

    # Input 4: 2D y with shape (M, 3), deg=3, full=True, cov=False
    M = 15
    x = np.linspace(-2.0, 2.0, M, dtype=np.float64)
    y = np.column_stack([x**3, x**2 + x, -0.5 * x**3 + 2.0]).astype(
        np.float64
    )
    w = np.linspace(0.8, 1.2, M, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 3,
            "rcond": 1e-7,
            "full": True,
            "w": w,
            "cov": False,
        }
    )

    # Input 5: 1D y, deg=2, full=False, cov=True
    M = 9
    x = np.array(
        [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32
    )
    y = (0.5 * x**2 - 1.2 * x + 4.0).astype(np.float32)
    w = np.ones(M, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 2,
            "rcond": 1e-4,
            "full": False,
            "w": w,
            "cov": True,
        }
    )

    # Input 6: 2D y with shape (M, 2), deg=2, full=False, cov=True
    M = 10
    x = np.linspace(0.5, 5.0, M, dtype=np.float64)
    y = np.column_stack([x**2 + 2.0, 3.0 * x**2 - x + 1.0]).astype(
        np.float64
    )
    w = np.ones(M, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 2,
            "rcond": 1e-6,
            "full": False,
            "w": w,
            "cov": True,
        }
    )

    # Input 7: 1D y, deg=1, full=True, cov=True (cov ignored when full=True)
    M = 6
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0], dtype=np.float32)
    w = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 1,
            "rcond": 1e-5,
            "full": True,
            "w": w,
            "cov": True,
        }
    )

    # Input 8: 2D y with shape (M, 1), deg=2, full=True, cov=True
    M = 7
    x = np.linspace(-3.0, 3.0, M, dtype=np.float32)
    y = (x**2).reshape(-1, 1).astype(np.float32)
    w = np.ones(M, dtype=np.float32)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 2,
            "rcond": 1e-5,
            "full": True,
            "w": w,
            "cov": True,
        }
    )

    # Input 9: deg=0 (constant fit), 1D y, full=False, cov=False
    M = 5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    y = np.array([3.1, 2.9, 3.0, 3.2, 2.8], dtype=np.float64)
    w = np.ones(M, dtype=np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 0,
            "rcond": 1e-6,
            "full": False,
            "w": w,
            "cov": False,
        }
    )

    # Input 10: deg=4, higher order fit with larger M, 2D y
    M = 20
    x = np.linspace(-1.5, 1.5, M, dtype=np.float64)
    y = np.column_stack([x**4 - 2 * x**2, 0.5 * x**4 + x**3]).astype(
        np.float64
    )
    w = np.random.uniform(0.9, 1.1, size=M).astype(np.float64)
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "deg": 4,
            "rcond": 1e-8,
            "full": False,
            "w": w,
            "cov": True,
        }
    )

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
