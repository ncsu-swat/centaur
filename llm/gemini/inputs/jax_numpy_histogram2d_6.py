
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic floats, density=False
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 5,
        "range": ((0.0, 10.0), (0.0, 10.0)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values, density=True
    x = np.array([-1.5, -2.5, 0.0, 1.5, 2.5], dtype=np.float32)
    y = np.array([-2.5, -1.5, 0.0, 2.5, 1.5], dtype=np.float32)
    weights = np.array([0.5, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 3,
        "range": ((-3.0, 3.0), (-3.0, 3.0)),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32 arrays, large scale
    x = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    y = np.array([60, 50, 40, 30, 20, 10], dtype=np.int32)
    weights = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 10,
        "range": ((0.0, 100.0), (0.0, 100.0)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large random datasets, float64
    x = np.random.uniform(-10, 10, size=(100,)).astype(np.float64)
    y = np.random.uniform(-10, 10, size=(100,)).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, size=(100,)).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 20,
        "range": ((-10.0, 10.0), (-10.0, 10.0)),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Skewed range, float32
    x = np.random.normal(0, 1, size=50).astype(np.float32)
    y = np.random.normal(5, 2, size=50).astype(np.float32)
    weights = np.ones(50, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 8,
        "range": ((-5.0, 5.0), (0.0, 10.0)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer types for x and y, int weights
    x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    y = np.array([5, 4, 3, 2, 1], dtype=np.int64)
    weights = np.array([1, 1, 1, 1, 1], dtype=np.int64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 4,
        "range": ((1.0, 5.0), (1.0, 5.0)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small number of bins, density=True
    x = np.array([0.1, 0.2, 0.9], dtype=np.float32)
    y = np.array([0.9, 0.8, 0.1], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 2,
        "range": ((0.0, 1.0), (0.0, 1.0)),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Highly asymmetric ranges
    x = np.random.rand(80).astype(np.float32) * 100.0
    y = np.random.rand(80).astype(np.float32) * 0.01
    weights = np.random.rand(80).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 12,
        "range": ((0.0, 100.0), (0.0, 0.01)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1 element arrays (edge case but valid)
    x = np.array([1.5], dtype=np.float32)
    y = np.array([2.5], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 1,
        "range": ((1.0, 2.0), (2.0, 3.0)),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float16 types
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    y = np.array([1.5, 2.5, 3.5], dtype=np.float16)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 3,
        "range": ((0.0, 3.0), (1.0, 4.0)),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_6"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_6'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_6'], lib="jax", suffix=6)
