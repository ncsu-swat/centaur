
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    a = np.random.randn(100).astype(np.float32)
    weights = np.random.rand(100).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 10,
        "range": (-3.0, 3.0),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive values, density=True
    a = np.random.rand(50).astype(np.float32) * 10.0
    weights = np.random.rand(50).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 5,
        "range": (0.0, 10.0),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array (will be flattened)
    a = np.random.randn(10, 10).astype(np.float32)
    weights = np.random.rand(10, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 15,
        "range": (-4.0, 4.0),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int32 array with float32 weights
    a = np.random.randint(0, 100, size=(80,)).astype(np.int32)
    weights = np.random.rand(80).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 8,
        "range": (0, 100),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 array, 50 bins
    a = np.random.randn(1000).astype(np.float64)
    weights = np.random.rand(1000).astype(np.float64)
    input_dict = {
        "a": a,
        "bins": 50,
        "range": (-5.0, 5.0),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array
    a = np.random.rand(5, 5, 4).astype(np.float32) * 5.0
    weights = np.random.rand(5, 5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 20,
        "range": (0.0, 5.0),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 1D array
    a = np.random.uniform(-10.0, 10.0, 200).astype(np.float64)
    weights = np.random.uniform(0.1, 1.0, 200).astype(np.float64)
    input_dict = {
        "a": a,
        "bins": 12,
        "range": (-10.0, 10.0),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 1D array with manual values
    a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "bins": 4,
        "range": (1.0, 6.0),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large range with negative values
    a = np.random.uniform(-100.0, 100.0, 500).astype(np.float32)
    weights = np.random.rand(500).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 40,
        "range": (-100.0, 100.0),
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 array with float32 weights
    a = np.random.randint(-50, 50, size=(150,)).astype(np.int16)
    weights = np.random.rand(150).astype(np.float32)
    input_dict = {
        "a": a,
        "bins": 25,
        "range": (-50, 50),
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram_1"] = histogram_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_1'.")


check_valid('jax.numpy.histogram', generated_inputs['jax.numpy.histogram_1'], lib="jax", suffix=1)
