
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard values
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 0D arrays (scalars as tensors)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64, with broadcasting (loc and scale are 1D)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    loc = np.array([0.5, 1.5], dtype=np.float64)
    scale = np.array([1.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: Negative values for x and loc, positive scale
    x = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    loc = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    scale = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Large scale values, 1D
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: 3D arrays, float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Very small positive scale values
    x = np.array([1.0, 1.0], dtype=np.float32)
    loc = np.array([0.9, 0.9], dtype=np.float32)
    scale = np.array([1e-3, 1e-3], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Broadcasting scale (scalar tensor) and loc (1D) to x (2D)
    x = np.array([[1.0, -1.0], [0.0, 2.0]], dtype=np.float32)
    loc = np.array([0.0, 1.0], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large negative x values
    x = np.array([-100.0, -50.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: High dimensions (4D) with positive random scale
    x = np.random.randn(2, 2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float64)
    scale = np.abs(np.random.randn(2, 2, 2, 2)).astype(np.float64) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logcdf_1"] = gumbel_r_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logcdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logcdf_1'.")


check_valid('jax.scipy.stats.gumbel_r.logcdf', generated_inputs['jax.scipy.stats.gumbel_r.logcdf_1'], lib="jax", suffix=1)
