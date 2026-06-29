
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gamma_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard values
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    loc = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scale = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 2: 0D arrays (scalars as numpy arrays)
    x = np.array(2.5, dtype=np.float32)
    a = np.array(3.0, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    a = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 4: Broadcasting shapes
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # shape (2, 3)
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)                    # shape (3,)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)                   # shape (2, 1)
    scale = np.array(2.0, dtype=np.float32)                            # shape ()
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 5: Negative values for loc and x (x < loc gives -inf logcdf, valid)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 6: Very small shape parameter a, float64
    x = np.array([0.1, 0.5, 1.0], dtype=np.float64)
    a = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 7: Large scale parameter
    x = np.array([100.0, 200.0], dtype=np.float32)
    a = np.array([2.0, 3.0], dtype=np.float32)
    loc = np.array([10.0, 20.0], dtype=np.float32)
    scale = np.array([50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 8: High dimensional 3D arrays
    x = np.random.uniform(5.0, 10.0, size=(2, 2, 2)).astype(np.float32)
    a = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-2.0, 2.0, size=(2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 3.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 9: Large values for shape a
    x = np.array([50.0], dtype=np.float64)
    a = np.array([40.0], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 10: Mixed shapes and dimensions (broadcasting)
    x = np.array([[[1.0], [2.0]]], dtype=np.float32)  # shape (1, 2, 1)
    a = np.array([[1.0, 2.0]], dtype=np.float32)      # shape (1, 2)
    loc = np.array([0.0], dtype=np.float32)           # shape (1,)
    scale = np.array(0.5, dtype=np.float32)           # shape ()
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.scipy.stats.gamma.logcdf"] = gamma_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.logcdf'.")


check_valid('jax.scipy.stats.gamma.logcdf', generated_inputs['jax.scipy.stats.gamma.logcdf'], lib="jax", suffix=0)
