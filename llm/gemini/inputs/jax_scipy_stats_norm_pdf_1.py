
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_norm_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays with non-zero loc and scale
    x = np.array([[-2.0, -1.0], [1.0, 2.0]], dtype=np.float32)
    loc = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    scale = np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: float64 precision 1D arrays
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([-0.1, -0.2, -0.3], dtype=np.float64)
    scale = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 0D arrays (scalars wrapped as tensors)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting 2D 'x' with 1D 'loc' and 1D 'scale'
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: 3D float32 arrays
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.5
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Small scale values (narrow distribution)
    x = np.array([-0.01, 0.0, 0.01], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Large scale values (broad distribution)
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    scale = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Negative mean/location parameters
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    loc = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: High-dimensional 4D arrays
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    loc = np.zeros((2, 3, 4, 5), dtype=np.float32)
    scale = np.ones((2, 3, 4, 5), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.norm.pdf_1"] = jax_scipy_stats_norm_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.pdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.pdf_1'.")


check_valid('jax.scipy.stats.norm.pdf', generated_inputs['jax.scipy.stats.norm.pdf_1'], lib="jax", suffix=1)
