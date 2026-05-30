
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def interp_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    xp = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    fp = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    left = np.array(-1.0, dtype=np.float32)
    right = np.array(5.0, dtype=np.float32)
    period = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 2: 2D x array with float64
    x = np.array([[0.5, 1.5], [1.0, 2.0]], dtype=np.float64)
    xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    fp = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    left = np.array(-99.0, dtype=np.float64)
    right = np.array(99.0, dtype=np.float64)
    period = np.array(360.0, dtype=np.float64)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 3: 3D x array with negative values in xp and fp
    x = np.random.uniform(-5.0, 5.0, (2, 2, 2)).astype(np.float32)
    xp = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    fp = np.array([-9.0, -1.0, 1.0, 9.0], dtype=np.float32)
    left = np.array(-10.0, dtype=np.float32)
    right = np.array(10.0, dtype=np.float32)
    period = np.array(20.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 4: Scalar-like 0D x array
    x = np.array(1.5, dtype=np.float32)
    xp = np.array([1.0, 2.0], dtype=np.float32)
    fp = np.array([10.0, 20.0], dtype=np.float32)
    left = np.array(5.0, dtype=np.float32)
    right = np.array(25.0, dtype=np.float32)
    period = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 5: Sine wave interpolation (larger 1D arrays)
    x = np.linspace(-10.0, 10.0, 100).astype(np.float32)
    xp = np.linspace(-5.0, 5.0, 50).astype(np.float32)
    fp = np.sin(xp).astype(np.float32)
    left = np.array(-1.0, dtype=np.float32)
    right = np.array(1.0, dtype=np.float32)
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 6: 4D x array with small values
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    xp = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    fp = np.array([4.0, 1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    left = np.array(0.0, dtype=np.float32)
    right = np.array(0.0, dtype=np.float32)
    period = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 7: Very small step values in float64
    x = np.array([1e-5, 2e-5], dtype=np.float64)
    xp = np.array([0.0, 1e-4], dtype=np.float64)
    fp = np.array([0.0, 1.0], dtype=np.float64)
    left = np.array(-1.0, dtype=np.float64)
    right = np.array(2.0, dtype=np.float64)
    period = np.array(1e-3, dtype=np.float64)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 8: High range values
    x = np.array([-100.0, 500.0], dtype=np.float32)
    xp = np.array([0.0, 400.0], dtype=np.float32)
    fp = np.array([0.0, 1600.0], dtype=np.float32)
    left = np.array(-400.0, dtype=np.float32)
    right = np.array(2000.0, dtype=np.float32)
    period = np.array(800.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 9: Non-symmetric extrapolation boundary values
    x = np.array([-2.5, 3.5], dtype=np.float32)
    xp = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    fp = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    left = np.array(-5.0, dtype=np.float32)
    right = np.array(5.0, dtype=np.float32)
    period = np.array(12.0, dtype=np.float32)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    # Input 10: Float64 high precision values
    x = np.array([0.3333333333333333], dtype=np.float64)
    xp = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    fp = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    left = np.array(0.0, dtype=np.float64)
    right = np.array(1.0, dtype=np.float64)
    period = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({
        'x': x, 'xp': xp, 'fp': fp, 'left': left, 'right': right, 'period': period
    })

    return list_of_inputs

generated_inputs["jax.numpy.interp_1"] = interp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.interp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.interp_1'.")


check_valid('jax.numpy.interp', generated_inputs['jax.numpy.interp_1'], lib="jax", suffix=1)
