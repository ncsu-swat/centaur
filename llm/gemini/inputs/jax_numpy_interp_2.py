
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def interp_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, periodic sine wave interpolation in float32
    xp = np.linspace(0.0, 2 * np.pi, 10, endpoint=False, dtype=np.float32)
    fp = np.sin(xp).astype(np.float32)
    x = np.array([0.5, 1.5, 3.0, 5.0], dtype=np.float32)
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 2: 2D x coordinates with float32 quadratic values
    xp = np.linspace(0.0, 10.0, 8, endpoint=False, dtype=np.float32)
    fp = (xp**2).astype(np.float32)
    x = np.array(
        [[-2.0, 1.0, 4.5, 9.0], [11.0, 3.0, 0.0, 7.5], [5.0, 2.2, 8.8, 12.0]],
        dtype=np.float32,
    )
    period = np.array(10.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 3: 3D x coordinates, float64 precision
    xp = np.linspace(0.0, 100.0, 15, endpoint=False, dtype=np.float64)
    fp = np.sqrt(xp).astype(np.float64)
    x = np.linspace(-50.0, 150.0, 30, dtype=np.float64).reshape((2, 3, 5))
    period = np.array(100.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 4: Scalar (0D) x coordinate
    xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    fp = np.array([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    x = np.array(3.5, dtype=np.float32)
    period = np.array(5.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 5: 1D array with negative values in float64
    xp = np.linspace(0.0, 12.0, 7, endpoint=False, dtype=np.float64)
    fp = np.cos(xp).astype(np.float64)
    x = np.array([-15.0, -5.0, 0.0, 5.0, 15.0, 25.0], dtype=np.float64)
    period = np.array(12.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 6: 4D x coordinates
    xp = np.linspace(0.0, 50.0, 20, endpoint=False, dtype=np.float32)
    fp = np.linspace(-10.0, 10.0, 20, dtype=np.float32)
    x = np.random.default_rng(42).uniform(-20.0, 70.0, size=(2, 2, 2, 3)).astype(np.float32)
    period = np.array(50.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 7: Minimal length xp (2 points), 1D dense x
    xp = np.array([0.0, 1.0], dtype=np.float32)
    fp = np.array([10.0, 20.0], dtype=np.float32)
    x = np.linspace(-2.0, 4.0, 50, dtype=np.float32)
    period = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 8: 2D column vector x (shape 5, 1) in angular space [0, 360)
    xp = np.linspace(0.0, 360.0, 50, endpoint=False, dtype=np.float64)
    fp = np.sin(np.radians(xp)).astype(np.float64)
    x = np.array([[45.0], [90.0], [180.0], [270.0], [400.0]], dtype=np.float64)
    period = np.array(360.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 9: Single-element 1D array x
    xp = np.array([1.0, 3.0, 7.0, 15.0], dtype=np.float32)
    fp = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    x = np.array([42.0], dtype=np.float32)
    period = np.array(20.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 10: 3D non-contiguous shaped x (shape 4, 1, 3) with exponential function
    xp = np.linspace(0.0, 1.0, 11, endpoint=False, dtype=np.float32)
    fp = np.exp(xp).astype(np.float32)
    x = np.linspace(-1.0, 2.0, 12, dtype=np.float32).reshape((4, 1, 3))
    period = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    # Input 11: 2D row-dominant x (shape 2, 6) in float64 with linear function
    xp = np.linspace(0.0, 24.0, 25, endpoint=False, dtype=np.float64)
    fp = (xp * 0.5 - 3.0).astype(np.float64)
    x = np.array(
        [
            [-12.0, 0.0, 6.0, 12.0, 18.0, 24.0],
            [30.0, 36.0, -6.0, 2.5, 14.2, 26.8],
        ],
        dtype=np.float64,
    )
    period = np.array(24.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "xp": xp,
                "fp": fp,
                "left": "extrapolate",
                "right": "extrapolate",
                "period": period,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.interp_2"] = interp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.interp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.interp_2'.")


check_valid('jax.numpy.interp', generated_inputs['jax.numpy.interp_2'], lib="jax", suffix=2)
