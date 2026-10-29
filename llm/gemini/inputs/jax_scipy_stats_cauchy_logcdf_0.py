
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalars (tensors), float64
    x = np.array(1.5, dtype=np.float64)
    loc = np.array(-0.5, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, matching shapes
    x = np.array([[-3.0, -1.5], [0.5, 2.5]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D loc and scale with 2D x
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: 3D arrays with float64
    x = np.random.randn(2, 3, 4).astype(np.float64)
    loc = np.zeros((2, 3, 4), dtype=np.float64)
    scale = np.ones((2, 3, 4), dtype=np.float64) * 3.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Large negative values for x (testing tail)
    x = np.array([-100.0, -50.0, -10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Large positive values for x (testing upper tail)
    x = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    loc = np.array([1.0, -2.0, 5.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: Broadcasting scalar loc and scale against multi-dimensional x
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Varying scale parameters (small and large scale)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.01, 1.0, 100.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Column and row broadcasting (2D)
    x = np.linspace(-10.0, 10.0, 6).reshape(6, 1).astype(np.float32)
    loc = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.logcdf"] = cauchy_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logcdf'.")


check_valid('jax.scipy.stats.cauchy.logcdf', generated_inputs['jax.scipy.stats.cauchy.logcdf'], lib="jax", suffix=0)
