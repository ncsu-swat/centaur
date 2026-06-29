
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_pareto_logsf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.5, 2.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 2: Scalar-like 0D arrays with float32
    x = np.array(5.0, dtype=np.float32)
    b = np.array(3.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 3: 2D arrays with float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 4: float64 arrays with larger values
    x = np.array([100.0, 200.0], dtype=np.float64)
    b = np.array([0.5, 1.2], dtype=np.float64)
    loc = np.array([-10.0, -20.0], dtype=np.float64)
    scale = np.array([5.0, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 5: Negative loc values, x < loc + scale (survival function is 1, log survival is 0)
    x = np.array([0.5, 1.5], dtype=np.float32)
    b = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([1.0, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 6: Broadcasting 1D inputs with scalar-like inputs
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.5], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 7: Multidimensional broadcasting
    x = np.ones((2, 3), dtype=np.float32) * 5.0
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 8: High dimensional 3D arrays
    x = np.ones((2, 2, 2), dtype=np.float32) * 4.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 9: Large scale and offset values
    x = np.array([1000.0], dtype=np.float32)
    b = np.array([5.0], dtype=np.float32)
    loc = np.array([100.0], dtype=np.float32)
    scale = np.array([500.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 10: Mixed types (float32 and float64)
    x = np.array([10.0, 20.0], dtype=np.float32)
    b = np.array([1.1, 2.2], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.logsf"] = jax_scipy_stats_pareto_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logsf'.")


check_valid('jax.scipy.stats.pareto.logsf', generated_inputs['jax.scipy.stats.pareto.logsf'], lib="jax", suffix=0)
