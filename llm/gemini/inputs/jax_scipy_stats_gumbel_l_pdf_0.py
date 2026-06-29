
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays, positive scale
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, mixed sign x, non-zero loc
    x = np.array([[-2.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    loc = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: Float64 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float64)
    loc = np.ones((2, 3, 4), dtype=np.float64) * 0.5
    scale = np.ones((2, 3, 4), dtype=np.float64) * 1.5
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 0D arrays (scalar arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting, 1D loc/scale, 2D x
    x = np.random.randn(3, 3).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    scale = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Small scale (0.1), negative loc
    x = np.array([-10.0, -5.0, 0.0], dtype=np.float32)
    loc = np.array([-5.0, -5.0, -5.0], dtype=np.float32)
    scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Broadcasting with scalar arrays
    x = np.random.randn(4, 4).astype(np.float32)
    loc = np.array(2.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Larger dimensions float32
    x = np.random.randn(10, 10).astype(np.float32)
    loc = np.zeros((10, 10), dtype=np.float32)
    scale = np.ones((10, 10), dtype=np.float32) * 5.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Broadcasting complex shape
    x = np.random.randn(5, 1, 3).astype(np.float32)
    loc = np.random.randn(1, 4, 3).astype(np.float32)
    scale = np.abs(np.random.randn(5, 4, 1)).astype(np.float32) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Float64 precision with negative/positive mixed values
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.pdf"] = gumbel_l_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.pdf'.")


check_valid('jax.scipy.stats.gumbel_l.pdf', generated_inputs['jax.scipy.stats.gumbel_l.pdf'], lib="jax", suffix=0)
