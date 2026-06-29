
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D float32 arrays with some values below loc (negative or small x)
    x = np.array([[-1.0, 0.5], [2.0, 3.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 3D float64 arrays
    x = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float64)
    loc = np.zeros((2, 3, 4), dtype=np.float64)
    scale = np.random.uniform(0.1, 2.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 0D (scalar) arrays
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broad-casting with 1D scale and loc, 2D x
    x = np.random.uniform(1.0, 10.0, size=(3, 5)).astype(np.float32)
    loc = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Large 1D array with float64
    x = np.linspace(0.1, 100.0, 1000, dtype=np.float64)
    loc = np.full((1000,), 0.0, dtype=np.float64)
    scale = np.full((1000,), 10.0, dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: Negative loc values
    x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    scale = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Very small scale parameter values
    x = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([0.001, 0.001, 0.001], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: High-dimensional (4D) inputs
    x = np.random.uniform(5.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.ones((2, 2, 2, 2), dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 5.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Mixed positive and negative values for x, scale always positive
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.logcdf"] = expon_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.logcdf'.")


check_valid('jax.scipy.stats.expon.logcdf', generated_inputs['jax.scipy.stats.expon.logcdf'], lib="jax", suffix=0)
