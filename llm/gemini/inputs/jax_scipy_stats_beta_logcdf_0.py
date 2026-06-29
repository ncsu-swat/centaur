
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def beta_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (standard beta)
    input_dict = {
        "x": np.array(0.5, dtype=np.float32),
        "a": np.array(2.0, dtype=np.float32),
        "b": np.array(3.0, dtype=np.float32),
        "loc": np.array(0.0, dtype=np.float32),
        "scale": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays
    input_dict = {
        "x": np.array([0.1, 0.5, 0.9], dtype=np.float32),
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "b": np.array([3.0, 2.0, 1.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays
    input_dict = {
        "x": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        "a": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "b": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
        "loc": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "scale": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Shifted loc and scale (float64)
    input_dict = {
        "x": np.array([2.5, 3.0], dtype=np.float64),
        "a": np.array([2.0, 2.0], dtype=np.float64),
        "b": np.array([2.0, 2.0], dtype=np.float64),
        "loc": np.array([1.0, 2.0], dtype=np.float64),
        "scale": np.array([3.0, 2.0], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Shape parameters < 1.0
    input_dict = {
        "x": np.array([0.25, 0.75], dtype=np.float32),
        "a": np.array([0.5, 0.2], dtype=np.float32),
        "b": np.array([0.5, 0.8], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large scale and shifted negative loc
    input_dict = {
        "x": np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        "a": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "b": np.array([5.0, 5.0, 5.0], dtype=np.float32),
        "loc": np.array([-20.0, -20.0, -20.0], dtype=np.float32),
        "scale": np.array([50.0, 50.0, 50.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays
    x_3d = np.random.uniform(0.1, 0.9, size=(2, 2, 2)).astype(np.float32)
    a_3d = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    b_3d = np.random.uniform(1.0, 5.0, size=(2, 2, 2)).astype(np.float32)
    loc_3d = np.zeros((2, 2, 2), dtype=np.float32)
    scale_3d = np.ones((2, 2, 2), dtype=np.float32)
    input_dict = {
        "x": x_3d,
        "a": a_3d,
        "b": b_3d,
        "loc": loc_3d,
        "scale": scale_3d
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Symmetric beta distribution shapes
    input_dict = {
        "x": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "a": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        "b": np.array([4.0, 4.0, 4.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large shape parameters
    input_dict = {
        "x": np.array([0.5], dtype=np.float32),
        "a": np.array([150.0], dtype=np.float32),
        "b": np.array([150.0], dtype=np.float32),
        "loc": np.array([0.0], dtype=np.float32),
        "scale": np.array([1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Value x outside of [loc, loc+scale] (e.g. x < loc and x > loc+scale)
    input_dict = {
        "x": np.array([-1.0, 2.0], dtype=np.float32),
        "a": np.array([2.0, 2.0], dtype=np.float32),
        "b": np.array([3.0, 3.0], dtype=np.float32),
        "loc": np.array([0.0, 0.0], dtype=np.float32),
        "scale": np.array([1.0, 1.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.beta.logcdf"] = beta_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.beta.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.beta.logcdf'.")


check_valid('jax.scipy.stats.beta.logcdf', generated_inputs['jax.scipy.stats.beta.logcdf'], lib="jax", suffix=0)
