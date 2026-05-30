
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_standardize_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, axis (1,), algorithm "fast"
    x = np.random.randn(4, 5).astype(np.float32)
    axis = (1,)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-5,
        "where": where,
        "algorithm": "fast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float32, axis (0, 2), algorithm "stable"
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-6,
        "where": where,
        "algorithm": "stable"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64, axis (0,), algorithm "fast"
    x = np.random.randn(10).astype(np.float64)
    axis = (0,)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-4,
        "where": where,
        "algorithm": "fast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32, axis (0, 1), algorithm "stable"
    x = np.random.randn(3, 3).astype(np.float32)
    axis = (0, 1)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-5,
        "where": where,
        "algorithm": "stable"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32, axis (1, 3), algorithm "fast"
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (1, 3)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-5,
        "where": where,
        "algorithm": "fast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32, axis (-1,), algorithm "stable"
    x = np.random.randn(5, 5).astype(np.float32)
    axis = (-1,)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-3,
        "where": where,
        "algorithm": "stable"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32, axis (2,), algorithm "fast"
    x = np.random.randn(8, 4, 16).astype(np.float32)
    axis = (2,)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-5,
        "where": where,
        "algorithm": "fast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64, axis (0, 2), algorithm "stable"
    x = np.random.randn(3, 1, 5).astype(np.float64)
    axis = (0, 2)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-6,
        "where": where,
        "algorithm": "stable"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32, axis (0,), algorithm "fast"
    x = np.random.randn(6, 6).astype(np.float32)
    axis = (0,)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 2e-5,
        "where": where,
        "algorithm": "fast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32, axis (0, 1, 2, 3), algorithm "stable"
    x = np.random.randn(2, 3, 2, 3).astype(np.float32)
    axis = (0, 1, 2, 3)
    where = np.ones_like(x, dtype=bool)
    mean = np.mean(x, axis=axis, keepdims=True, where=where)
    variance = np.var(x, axis=axis, keepdims=True, where=where)
    input_dict = {
        "x": x,
        "axis": axis,
        "mean": mean,
        "variance": variance,
        "epsilon": 1e-5,
        "where": where,
        "algorithm": "stable"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.standardize_2"] = jax_nn_standardize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.standardize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.standardize_2'.")


check_valid('jax.nn.standardize', generated_inputs['jax.nn.standardize_2'], lib="jax", suffix=2)
