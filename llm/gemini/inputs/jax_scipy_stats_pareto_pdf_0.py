
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pareto_pdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, simple float32
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 2: Scalar arrays (0D tensors), float32
    x = np.array(2.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64
    x = np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float64)
    b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 4: Broadcasting shapes
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([[1.5], [2.5]], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 5: Offset and non-default scale, 1D
    x = np.array([5.0, 6.0], dtype=np.float32)
    b = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([1.0, 2.0], dtype=np.float32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 6: 3D arrays, float32 with randomized positive values
    x = (np.random.rand(2, 2, 2) + 2.0).astype(np.float32)
    b = (np.random.rand(2, 2, 2) + 0.5).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 7: Large scale and shift, float64
    x = np.array([100.0, 200.0], dtype=np.float64)
    b = np.array([3.5, 4.5], dtype=np.float64)
    loc = np.array([-10.0, -20.0], dtype=np.float64)
    scale = np.array([10.0, 20.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 8: Small 1D arrays, float32, small shapes
    x = np.array([1.5], dtype=np.float32)
    b = np.array([0.5], dtype=np.float32)
    loc = np.array([-1.0], dtype=np.float32)
    scale = np.array([0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 9: 4D tensors, float32
    x = (np.ones((2, 2, 2, 2)) * 3.0).astype(np.float32)
    b = (np.ones((2, 2, 2, 2)) * 1.2).astype(np.float32)
    loc = (np.ones((2, 2, 2, 2)) * 0.5).astype(np.float32)
    scale = (np.ones((2, 2, 2, 2)) * 1.5).astype(np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    # Input 10: Sharp distribution parameters (high b), float32
    x = np.array([1.1, 1.2, 1.3], dtype=np.float32)
    b = np.array([10.0, 15.0, 20.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "b": b, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.pdf"] = pareto_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.pdf'.")


check_valid('jax.scipy.stats.pareto.pdf', generated_inputs['jax.scipy.stats.pareto.pdf'], lib="jax", suffix=0)
