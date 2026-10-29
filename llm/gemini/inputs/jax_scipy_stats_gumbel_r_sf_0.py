
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_sf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar tensors (float32)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 1D tensors
    x = np.array([-2.0, -0.5, 0.0, 1.2, 3.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 1.5, 2.0, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D tensors (float64)
    x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float64)
    loc = np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting 1D to 2D
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: Broadcasting across different dimensions (2D with column/row vectors)
    x = np.linspace(-3.0, 3.0, 5, dtype=np.float32).reshape(5, 1)
    loc = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: 3D tensors
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Large negative and positive values of x
    x = np.array([-100.0, -10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: Small scale values with float64 precision
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    scale = np.array([1e-3, 1e-2, 1e-1], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: 4D tensor inputs
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: Broadcasting scalar scale and loc to multi-dimensional x
    x = np.random.randn(4, 5).astype(np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

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
