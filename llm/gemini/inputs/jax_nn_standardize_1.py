
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_standardize_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=0, fast algorithm
    x = np.random.randn(10).astype(np.float32)
    axis = 0
    mean = np.array([0.0], dtype=np.float32)
    variance = np.array([1.0], dtype=np.float32)
    epsilon = 1e-5
    where = np.ones_like(x, dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 2: 2D array, axis=-1, stable algorithm, small epsilon
    x = np.random.randn(5, 10).astype(np.float32)
    axis = -1
    mean = np.random.randn(5, 1).astype(np.float32)
    variance = np.abs(np.random.randn(5, 1)).astype(np.float32) + 0.1
    epsilon = 1e-6
    where = np.random.choice([True, False], size=x.shape, p=[0.9, 0.1])
    algorithm = "stable"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 3: 2D array, axis=0, float64, fast algorithm
    x = np.random.randn(8, 6).astype(np.float64)
    axis = 0
    mean = np.random.randn(1, 6).astype(np.float64)
    variance = np.abs(np.random.randn(1, 6)).astype(np.float64) + 0.1
    epsilon = 1e-5
    where = np.ones_like(x, dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 4: 3D array, axis=1, stable algorithm
    x = np.random.randn(2, 4, 3).astype(np.float32)
    axis = 1
    mean = np.random.randn(2, 1, 3).astype(np.float32)
    variance = np.abs(np.random.randn(2, 1, 3)).astype(np.float32) + 0.1
    epsilon = 1e-4
    where = np.ones_like(x, dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 5: 3D array, axis=-1, larger epsilon, fast algorithm
    x = np.random.randn(3, 3, 5).astype(np.float32)
    axis = -1
    mean = np.random.randn(3, 3, 1).astype(np.float32)
    variance = np.abs(np.random.randn(3, 3, 1)).astype(np.float32) + 0.1
    epsilon = 1e-3
    where = np.random.choice([True, False], size=x.shape, p=[0.8, 0.2])
    algorithm = "fast"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 6: 4D array, axis=2, stable algorithm
    x = np.random.randn(2, 2, 4, 3).astype(np.float32)
    axis = 2
    mean = np.random.randn(2, 2, 1, 3).astype(np.float32)
    variance = np.abs(np.random.randn(2, 2, 1, 3)).astype(np.float32) + 0.1
    epsilon = 1e-5
    where = np.ones_like(x, dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 7: 1D array with negative values, float64, axis=0
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    axis = 0
    mean = np.array([0.0], dtype=np.float64)
    variance = np.array([50.0], dtype=np.float64)
    epsilon = 1e-8
    where = np.ones_like(x, dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 8: 2D array, axis=1, high variance
    x = np.random.randn(4, 4).astype(np.float32) * 100.0
    axis = 1
    mean = np.zeros((4, 1), dtype=np.float32)
    variance = np.ones((4, 1), dtype=np.float32) * 10000.0
    epsilon = 1e-5
    where = np.ones_like(x, dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 9: 3D array, axis=0, stable algorithm, custom where mask
    x = np.random.randn(4, 2, 2).astype(np.float32)
    axis = 0
    mean = np.random.randn(1, 2, 2).astype(np.float32)
    variance = np.abs(np.random.randn(1, 2, 2)).astype(np.float32) + 0.2
    epsilon = 1e-5
    where = np.array([
        [[True, False], [True, True]],
        [[False, True], [True, False]],
        [[True, True], [False, True]],
        [[True, False], [True, True]]
    ], dtype=bool)
    algorithm = "stable"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    # Input 10: 4D array, axis=-1, float64, fast algorithm
    x = np.random.randn(2, 2, 2, 2).astype(np.float64)
    axis = -1
    mean = np.random.randn(2, 2, 2, 1).astype(np.float64)
    variance = np.abs(np.random.randn(2, 2, 2, 1)).astype(np.float64) + 0.05
    epsilon = 1e-6
    where = np.ones_like(x, dtype=bool)
    algorithm = "fast"
    list_of_inputs.append({
        "x": x, "axis": axis, "mean": mean, "variance": variance,
        "epsilon": epsilon, "where": where, "algorithm": algorithm
    })

    return list_of_inputs

generated_inputs["jax.nn.standardize_1"] = jax_nn_standardize_inputs()

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
