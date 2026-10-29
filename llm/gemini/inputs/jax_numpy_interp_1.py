
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def interp_inputs():
    list_of_inputs = []

    # Input 1: 1D x, standard float32 coordinates within range
    xp = np.linspace(0.0, 10.0, 20, dtype=np.float32)
    fp = np.sin(xp).astype(np.float32)
    x = np.array([0.5, 2.3, 5.0, 8.7], dtype=np.float32)
    left = np.array(0.0, dtype=np.float32)
    right = np.array(0.0, dtype=np.float32)
    period = np.array(20.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 2: 2D x with float64 precision
    xp = np.linspace(-5.0, 5.0, 50, dtype=np.float64)
    fp = (xp ** 2).astype(np.float64)
    x = np.array([[-3.5, -1.0], [2.0, 4.5]], dtype=np.float64)
    left = np.array(-999.0, dtype=np.float64)
    right = np.array(999.0, dtype=np.float64)
    period = np.array(30.0, dtype=np.float64)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 3: Scalar (0-D) x
    xp = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    x = np.array(1.5, dtype=np.float32)
    left = np.array(10.0, dtype=np.float32)
    right = np.array(40.0, dtype=np.float32)
    period = np.array(10.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 4: 3D x tensor
    xp = np.linspace(1.0, 100.0, 100, dtype=np.float32)
    fp = np.log(xp).astype(np.float32)
    x = np.random.uniform(1.0, 100.0, size=(2, 3, 4)).astype(np.float32)
    left = np.array(0.0, dtype=np.float32)
    right = np.array(5.0, dtype=np.float32)
    period = np.array(200.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 5: Minimal xp and fp (2 elements)
    xp = np.array([0.0, 1.0], dtype=np.float32)
    fp = np.array([5.0, 15.0], dtype=np.float32)
    x = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    left = np.array(5.0, dtype=np.float32)
    right = np.array(15.0, dtype=np.float32)
    period = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 6: Strictly negative coordinates and values
    xp = np.linspace(-100.0, -10.0, 30, dtype=np.float64)
    fp = np.linspace(-50.0, -5.0, 30, dtype=np.float64)
    x = np.array([-80.0, -50.0, -20.0], dtype=np.float64)
    left = np.array(-50.0, dtype=np.float64)
    right = np.array(-5.0, dtype=np.float64)
    period = np.array(150.0, dtype=np.float64)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 7: Float16 dtype inputs
    xp = np.linspace(0.0, 4.0, 16, dtype=np.float16)
    fp = (xp * 3.0).astype(np.float16)
    x = np.array([1.25, 2.75, 3.5], dtype=np.float16)
    left = np.array(0.0, dtype=np.float16)
    right = np.array(12.0, dtype=np.float16)
    period = np.array(10.0, dtype=np.float16)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 8: 4D x array with large interpolation grid
    xp = np.linspace(0.0, 1.0, 500, dtype=np.float32)
    fp = np.exp(-xp).astype(np.float32)
    x = np.random.uniform(0.0, 1.0, size=(1, 2, 2, 3)).astype(np.float32)
    left = np.array(1.0, dtype=np.float32)
    right = np.array(np.exp(-1.0), dtype=np.float32)
    period = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 9: Non-uniform spacing in xp
    xp = np.array([0.0, 0.1, 0.5, 2.0, 10.0], dtype=np.float64)
    fp = np.array([1.0, 2.0, 3.0, 5.0, 8.0], dtype=np.float64)
    x = np.array([0.05, 0.3, 1.0, 6.0], dtype=np.float64)
    left = np.array(1.0, dtype=np.float64)
    right = np.array(8.0, dtype=np.float64)
    period = np.array(20.0, dtype=np.float64)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    # Input 10: Angular space periodic interpolation
    xp = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float32)
    fp = np.sin(xp).astype(np.float32)
    x = np.array([0.0, np.pi / 4, 2 * np.pi, 5 * np.pi / 2], dtype=np.float32)
    left = np.array(0.0, dtype=np.float32)
    right = np.array(0.0, dtype=np.float32)
    period = np.array(2 * np.pi, dtype=np.float32)
    list_of_inputs.append({
        "x": x, "xp": xp, "fp": fp, "left": left, "right": right, "period": period
    })

    return [copy.deepcopy(inp) for inp in list_of_inputs]

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
