
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_pareto_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32, valid support
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays
    x = np.array([[2.0, 2.5], [3.0, 3.5]], dtype=np.float32)
    b = np.array([[1.5, 2.5], [0.5, 1.0]], dtype=np.float32)
    loc = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 3: 1D float64 arrays
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    b = np.array([2.5, 3.5, 4.5], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 4: 0D arrays (scalars wrapped in arrays)
    x = np.array(5.0, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 5: Some elements out of support (x < loc + scale) to test behavior
    x = np.array([0.5, 2.0, 5.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 6: 3D float32 arrays
    x = np.ones((2, 2, 2), dtype=np.float32) * 4.0
    b = np.ones((2, 2, 2), dtype=np.float32) * 1.5
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 7: Small positive b shape parameter
    x = np.array([1.5, 2.5], dtype=np.float32)
    b = np.array([0.01, 0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 8: Broadcasting dimensions (3, 1) and (1, 3)
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    loc = np.zeros((3, 3), dtype=np.float32)
    scale = np.ones((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 9: Large parameter values
    x = np.array([100.0, 200.0], dtype=np.float64)
    b = np.array([10.0, 20.0], dtype=np.float64)
    loc = np.array([10.0, 20.0], dtype=np.float64)
    scale = np.array([5.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 10: Mixed float64 multi-dimensional inputs with varying offset/scales
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    b = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64)
    loc = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    scale = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.logpdf"] = jax_scipy_stats_pareto_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logpdf'.")


check_valid('jax.scipy.stats.pareto.logpdf', generated_inputs['jax.scipy.stats.pareto.logpdf'], lib="jax", suffix=0)
