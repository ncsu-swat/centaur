
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_sf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: Scalar (0D) float64 arrays
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D arrays with broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0, 0.5]], dtype=np.float32)  # shape (1, 2)
    scale = np.array([[1.5], [2.5]], dtype=np.float32)  # shape (2, 1)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Large scale values
    x = np.linspace(-10.0, 10.0, 5, dtype=np.float32)
    loc = np.zeros(5, dtype=np.float32)
    scale = np.ones(5, dtype=np.float32) * 10.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Very small scale values
    x = np.array([0.1, 0.2], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([0.01, 0.05], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Random 3D arrays
    x = np.random.randn(2, 3, 2).astype(np.float32)
    loc = np.random.randn(2, 3, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 2)).astype(np.float32) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Broadcasting 1D to larger size
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Random float64 2D matrices
    x = np.random.uniform(-5.0, 5.0, (4, 4)).astype(np.float64)
    loc = np.random.uniform(-2.0, 2.0, (4, 4)).astype(np.float64)
    scale = np.random.uniform(0.5, 3.0, (4, 4)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large positive x values (resulting in survival function close to 0)
    x = np.array([50.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Extreme negative x values (resulting in survival function close to 1)
    x = np.array([-50.0, -100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 11: 4D arrays
    x = np.random.normal(size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2)).astype(np.float32)
    scale = np.ones((2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.sf"] = gumbel_r_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.sf'.")


check_valid('jax.scipy.stats.gumbel_r.sf', generated_inputs['jax.scipy.stats.gumbel_r.sf'], lib="jax", suffix=0)
