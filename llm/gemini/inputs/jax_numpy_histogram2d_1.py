
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, positive, density=False
    x1 = np.random.rand(100).astype(np.float32) * 10
    y1 = np.random.rand(100).astype(np.float32) * 10
    w1 = np.random.rand(100).astype(np.float32)
    input_dict1 = {
        'x': x1,
        'y': y1,
        'bins': 10,
        'range': [[0.0, 10.0], [0.0, 10.0]],
        'weights': w1,
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative and positive values, float64, density=True
    x2 = (np.random.rand(50) * 20 - 10).astype(np.float64)
    y2 = (np.random.rand(50) * 20 - 10).astype(np.float64)
    w2 = np.random.rand(50).astype(np.float64)
    input_dict2 = {
        'x': x2,
        'y': y2,
        'bins': 5,
        'range': [[-10.0, 10.0], [-10.0, 10.0]],
        'weights': w2,
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Larger arrays, float32, density=False, 20 bins
    x3 = np.random.randn(1000).astype(np.float32)
    y3 = np.random.randn(1000).astype(np.float32)
    w3 = np.random.randn(1000).astype(np.float32)
    input_dict3 = {
        'x': x3,
        'y': y3,
        'bins': 20,
        'range': [[-3.0, 3.0], [-3.0, 3.0]],
        'weights': w3,
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Small array size, density=True
    x4 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y4 = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    w4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    input_dict4 = {
        'x': x4,
        'y': y4,
        'bins': 3,
        'range': [[1.0, 5.0], [1.0, 5.0]],
        'weights': w4,
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Int values for x, y, and weights
    x5 = np.random.randint(0, 50, size=200).astype(np.int32)
    y5 = np.random.randint(0, 50, size=200).astype(np.int32)
    w5 = np.ones(200, dtype=np.int32)
    input_dict5 = {
        'x': x5,
        'y': y5,
        'bins': 8,
        'range': [[0, 50], [0, 50]],
        'weights': w5,
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Uniform values, floats, density=True
    x6 = np.ones(10, dtype=np.float32) * 5.0
    y6 = np.ones(10, dtype=np.float32) * 5.0
    w6 = np.random.rand(10).astype(np.float32)
    input_dict6 = {
        'x': x6,
        'y': y6,
        'bins': 2,
        'range': [[0.0, 10.0], [0.0, 10.0]],
        'weights': w6,
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Wide range, float64, 50 bins
    x7 = (np.random.rand(500) * 200 - 100).astype(np.float64)
    y7 = (np.random.rand(500) * 200 - 100).astype(np.float64)
    w7 = np.random.rand(500).astype(np.float64)
    input_dict7 = {
        'x': x7,
        'y': y7,
        'bins': 50,
        'range': [[-100.0, 100.0], [-100.0, 100.0]],
        'weights': w7,
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Very small floats, density=True
    x8 = (np.random.rand(150) * 1e-5).astype(np.float32)
    y8 = (np.random.rand(150) * 1e-5).astype(np.float32)
    w8 = (np.random.rand(150) * 1e-3).astype(np.float32)
    input_dict8 = {
        'x': x8,
        'y': y8,
        'bins': 4,
        'range': [[0.0, 1e-5], [0.0, 1e-5]],
        'weights': w8,
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Negative weights, float32, density=False
    x9 = np.random.rand(80).astype(np.float32)
    y9 = np.random.rand(80).astype(np.float32)
    w9 = (np.random.rand(80) - 0.5).astype(np.float32)
    input_dict9 = {
        'x': x9,
        'y': y9,
        'bins': 6,
        'range': [[0.0, 1.0], [0.0, 1.0]],
        'weights': w9,
        'density': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Single element arrays
    x10 = np.array([2.5], dtype=np.float32)
    y10 = np.array([3.5], dtype=np.float32)
    w10 = np.array([1.0], dtype=np.float32)
    input_dict10 = {
        'x': x10,
        'y': y10,
        'bins': 1,
        'range': [[0.0, 5.0], [0.0, 5.0]],
        'weights': w10,
        'density': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_1"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_1'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_1'], lib="jax", suffix=1)
