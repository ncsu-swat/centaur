
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_logpdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (float32)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D arrays with matching shapes
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 2.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: Float64 precision
    x = np.array([1.23456789, -0.98765432], dtype=np.float64)
    loc = np.array([-1.0, 2.0], dtype=np.float64)
    scale = np.array([0.5, 3.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 2D arrays
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0, 0.5], [0.8, 1.2, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting - 2D x with 1D loc and 0D scale
    x = np.ones((3, 4), dtype=np.float32) * 2.0
    loc = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Broadcasting - loc column vector, scale row vector
    x = np.zeros((4, 3), dtype=np.float64)
    loc = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: 3D tensors
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Integer arrays promoted to floats
    x = np.array([-3, -1, 0, 2, 5], dtype=np.int32)
    loc = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    scale = np.array([1, 2, 1, 3, 2], dtype=np.int32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large positive and negative values
    x = np.array([-50.0, -10.0, 10.0, 50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: 4D tensor with broadcasted scale
    x = np.ones((2, 1, 3, 2), dtype=np.float32)
    loc = np.zeros((2, 1, 3, 2), dtype=np.float32)
    scale = np.array([[[[2.0]]]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 11: Small scale values
    x = np.array([0.001, -0.002, 0.005], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.01, 0.05, 0.02], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_r.logpdf"] = gumbel_r_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logpdf'.")


check_valid('jax.scipy.stats.gumbel_r.logpdf', generated_inputs['jax.scipy.stats.gumbel_r.logpdf'], lib="jax", suffix=0)
