
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def standardize_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, fast algorithm, float32
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    axis = 0
    mean = np.array([0.0], dtype=np.float32)
    variance = np.array([2.0], dtype=np.float32)
    epsilon = 1e-5
    where = np.ones_like(x, dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 2: 2D array, axis=-1, stable algorithm, float32
    x = np.random.randn(4, 8).astype(np.float32)
    axis = -1
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-4
    where = (x > -1.0)
    algorithm = "stable"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 3: 2D array, axis=0, fast algorithm, float64
    x = np.random.randn(5, 3).astype(np.float64)
    axis = 0
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-6
    where = np.ones((5, 3), dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 4: 3D array, axis=1, stable algorithm, float32
    x = np.random.randn(2, 6, 4).astype(np.float32)
    axis = 1
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-5
    where = np.ones((2, 6, 4), dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 5: 3D array, axis=2, fast algorithm, float32
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axis = 2
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-3
    where = (x < 1.0)
    algorithm = "fast"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 6: 4D array, axis=-1, stable algorithm, float32
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = -1
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-5
    where = np.ones((2, 3, 4, 5), dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 7: 4D array, axis=0, fast algorithm, float64
    x = np.random.randn(4, 2, 2, 3).astype(np.float64)
    axis = 0
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-7
    where = np.ones((4, 2, 2, 3), dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 8: 1D array, axis=-1, negative values, stable algorithm, float32
    x = np.array([-10.0, -20.0, -30.0, -40.0], dtype=np.float32)
    axis = -1
    mean = np.array([-25.0], dtype=np.float32)
    variance = np.array([125.0], dtype=np.float32)
    epsilon = 1e-5
    where = np.array([True, True, True, True])
    algorithm = "stable"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 9: 3D array, axis=-2, fast algorithm, float32
    x = np.random.randn(2, 5, 3).astype(np.float32)
    axis = -2
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-5
    where = (x >= 0.0)
    algorithm = "fast"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    # Input 10: 5D array, axis=4, stable algorithm, float32
    x = np.random.randn(2, 2, 2, 2, 4).astype(np.float32)
    axis = 4
    mean = np.mean(x, axis=axis, keepdims=True)
    variance = np.var(x, axis=axis, keepdims=True)
    epsilon = 1e-5
    where = np.ones((2, 2, 2, 2, 4), dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": copy.deepcopy(x),
        "axis": axis,
        "mean": copy.deepcopy(mean),
        "variance": copy.deepcopy(variance),
        "epsilon": epsilon,
        "where": copy.deepcopy(where),
        "algorithm": algorithm
    })

    return list_of_inputs

generated_inputs["jax.nn.standardize_1"] = standardize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.standardize_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.standardize_1'.")


check_valid('jax.nn.standardize', generated_inputs['jax.nn.standardize_1'], lib="jax", suffix=1)
