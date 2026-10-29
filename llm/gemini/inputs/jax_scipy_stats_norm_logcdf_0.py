
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def norm_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays with standard normal distribution (loc=0, scale=1)
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 2: 0D scalar tensors (float64)
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 3: 2D arrays with identical shape
    x = np.array([[-2.0, -1.0], [0.5, 2.5]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 4: Broadcasting shapes: x (3, 1), loc (1, 4), scale (1, 1)
    x = np.linspace(-2.0, 2.0, 3, dtype=np.float32).reshape(3, 1)
    loc = np.linspace(-1.0, 1.0, 4, dtype=np.float32).reshape(1, 4)
    scale = np.array([[0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 5: 3D arrays with float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    loc = np.zeros((2, 3, 4), dtype=np.float64)
    scale = np.ones((2, 3, 4), dtype=np.float64) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 6: Large negative values (extreme tail)
    x = np.array([-100.0, -50.0, -20.0, -10.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 7: Large positive values (near zero logcdf)
    x = np.array([10.0, 20.0, 50.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 8: 4D tensors with broadcasting
    x = np.random.randn(2, 1, 3, 1).astype(np.float32)
    loc = np.random.randn(1, 4, 1, 5).astype(np.float32)
    scale = np.abs(np.random.randn(1, 1, 1, 1).astype(np.float32)) + 0.1
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 9: Integer tensors (promoted to floating-point during computation)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    loc = np.array([0], dtype=np.int32)
    scale = np.array([2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    # Input 10: Varied scale parameters (float64)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.1, 1.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "loc": loc, "scale": scale}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.norm.logcdf"] = norm_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.norm.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.norm.logcdf'.")


check_valid('jax.scipy.stats.norm.logcdf', generated_inputs['jax.scipy.stats.norm.logcdf'], lib="jax", suffix=0)
