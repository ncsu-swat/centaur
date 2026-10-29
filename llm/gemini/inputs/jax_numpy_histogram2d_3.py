
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D inputs with equal bins
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [5, 5],
        "range": [[0.0, 10.0], [0.0, 10.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 inputs with density=True
    x = np.random.uniform(-5.0, 5.0, size=100).astype(np.float64)
    y = np.random.uniform(-5.0, 5.0, size=100).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, size=100).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [10, 10],
        "range": [[-5.0, 5.0], [-5.0, 5.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Asymmetric bin counts and integer data types
    x = np.random.randint(0, 50, size=60).astype(np.int32)
    y = np.random.randint(0, 100, size=60).astype(np.int32)
    weights = np.random.rand(60).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [5, 10],
        "range": [[0, 50], [0, 100]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative to positive ranges with int64 inputs
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    y = np.array([-20, -10, 0, 10, 20], dtype=np.int64)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [4, 8],
        "range": [[-15.0, 15.0], [-25.0, 25.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single element arrays
    x = np.array([2.5], dtype=np.float32)
    y = np.array([3.5], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [2, 2],
        "range": [[0.0, 5.0], [0.0, 5.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger sample size with density=True
    x = np.random.normal(loc=0.0, scale=1.0, size=500).astype(np.float32)
    y = np.random.normal(loc=1.0, scale=2.0, size=500).astype(np.float32)
    weights = np.ones(500, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [20, 25],
        "range": [[-3.0, 3.0], [-5.0, 7.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniform weights with int32 weights
    x = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    y = np.linspace(10.0, 20.0, 40, dtype=np.float32)
    weights = np.ones(40, dtype=np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [8, 4],
        "range": [[0.0, 1.0], [10.0, 20.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative ranges strictly below zero
    x = np.random.uniform(-100.0, -10.0, size=80).astype(np.float64)
    y = np.random.uniform(-50.0, -5.0, size=80).astype(np.float64)
    weights = np.random.uniform(1.0, 2.0, size=80).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [10, 5],
        "range": [[-100.0, -10.0], [-50.0, -5.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Highly granular bins
    x = np.random.rand(250).astype(np.float32)
    y = np.random.rand(250).astype(np.float32)
    weights = np.random.rand(250).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [30, 30],
        "range": [[0.0, 1.0], [0.0, 1.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-square dimensions with float weights
    x = np.array([0.1, 0.4, 0.7, 1.2, 1.5, 1.8], dtype=np.float32)
    y = np.array([2.1, 2.4, 2.7, 3.2, 3.5, 3.8], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": [3, 6],
        "range": [[0.0, 2.0], [2.0, 4.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.histogram2d_3"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_3'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_3'], lib="jax", suffix=3)
