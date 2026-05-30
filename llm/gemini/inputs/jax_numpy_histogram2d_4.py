
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "x": np.random.randn(100).astype(np.float32),
        "y": np.random.randn(100).astype(np.float32),
        "bins": (10, 10),
        "range": [[-3.0, 3.0], [-3.0, 3.0]],
        "weights": np.ones(100, dtype=np.float32),
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "x": np.random.randn(200).astype(np.float32),
        "y": np.random.randn(200).astype(np.float32),
        "bins": (5, 8),
        "range": [[-2.5, 2.5], [-2.5, 2.5]],
        "weights": np.random.rand(200).astype(np.float32),
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "x": np.random.uniform(-10.0, 10.0, 50).astype(np.float64),
        "y": np.random.uniform(-10.0, 10.0, 50).astype(np.float64),
        "bins": (12, 12),
        "range": [[-10.0, 10.0], [-10.0, 10.0]],
        "weights": np.random.rand(50).astype(np.float64),
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "x": np.linspace(0, 10, 50).astype(np.float32),
        "y": np.linspace(0, 20, 50).astype(np.float32),
        "bins": (10, 20),
        "range": [[0.0, 10.0], [0.0, 20.0]],
        "weights": np.ones(50, dtype=np.float32),
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "x": np.random.exponential(scale=2.0, size=150).astype(np.float32),
        "y": np.random.exponential(scale=2.0, size=150).astype(np.float32),
        "bins": (15, 15),
        "range": [[0.0, 10.0], [0.0, 10.0]],
        "weights": np.random.uniform(0.5, 1.5, 150).astype(np.float32),
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "x": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "y": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        "bins": (np.array([1.0, 3.0, 5.0], dtype=np.float32), np.array([1.0, 3.0, 5.0], dtype=np.float32)),
        "range": [[1.0, 5.0], [1.0, 5.0]],
        "weights": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "x": np.random.normal(0, 1, 300).astype(np.float64),
        "y": np.random.normal(0, 1, 300).astype(np.float64),
        "bins": (20, 10),
        "range": [[-4.0, 4.0], [-4.0, 4.0]],
        "weights": np.ones(300, dtype=np.float64),
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "x": np.arange(10, dtype=np.float32),
        "y": np.arange(10, dtype=np.float32),
        "bins": (5, 5),
        "range": [[0.0, 9.0], [0.0, 9.0]],
        "weights": np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], dtype=np.float32),
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "x": np.random.uniform(-50, 50, 100).astype(np.int32).astype(np.float32),
        "y": np.random.uniform(-50, 50, 100).astype(np.int32).astype(np.float32),
        "bins": (10, 10),
        "range": [[-50.0, 50.0], [-50.0, 50.0]],
        "weights": np.ones(100, dtype=np.float32),
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "x": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        "y": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        "bins": (np.array([-1.0, 0.0, 1.0], dtype=np.float64), np.array([-1.0, 0.0, 1.0], dtype=np.float64)),
        "range": [[-1.0, 1.0], [-1.0, 1.0]],
        "weights": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_4"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_4'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_4'], lib="jax", suffix=4)
