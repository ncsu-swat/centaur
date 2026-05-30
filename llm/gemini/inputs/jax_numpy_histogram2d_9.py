
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic floats
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    bins = (2, 2)
    range_val = ((1.0, 4.0), (1.0, 4.0))
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 with negatives and density=True
    x = np.random.uniform(-5, 5, 100).astype(np.float64)
    y = np.random.uniform(-5, 5, 100).astype(np.float64)
    bins = (10, 10)
    range_val = ((-5.0, 5.0), (-5.0, 5.0))
    weights = np.ones(100, dtype=np.float64)
    density = True
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Specifying bin edges as tuple of arrays
    x = np.random.normal(0, 1, 50).astype(np.float32)
    y = np.random.normal(0, 1, 50).astype(np.float32)
    bins = (np.array([-3.0, 0.0, 3.0], dtype=np.float32), np.array([-3.0, 0.0, 3.0], dtype=np.float32))
    range_val = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.random.uniform(0, 1, 50).astype(np.float32)
    density = False
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer inputs, density=True
    x = np.arange(10).astype(np.int32)
    y = np.arange(10).astype(np.int32)
    bins = (5, 5)
    range_val = ((0.0, 10.0), (0.0, 10.0))
    weights = np.ones(10, dtype=np.float32)
    density = True
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Asymmetric bins, larger sizes
    x = np.linspace(-10, 10, 200).astype(np.float32)
    y = np.linspace(-10, 10, 200).astype(np.float32)
    bins = (8, 12)
    range_val = ((-10.0, 10.0), (-10.0, 10.0))
    weights = np.random.exponential(1.0, 200).astype(np.float32)
    density = False
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element arrays
    x = np.array([2.5], dtype=np.float32)
    y = np.array([2.5], dtype=np.float32)
    bins = (1, 1)
    range_val = ((0.0, 5.0), (0.0, 5.0))
    weights = np.array([2.0], dtype=np.float32)
    density = True
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uniform random variables, large dataset
    x = np.random.rand(1000).astype(np.float32)
    y = np.random.rand(1000).astype(np.float32)
    bins = (20, 20)
    range_val = ((0.0, 1.0), (0.0, 1.0))
    weights = np.random.rand(1000).astype(np.float32)
    density = False
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative ranges and coordinates
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    bins = (3, 3)
    range_val = ((-4.0, 0.0), (-4.0, 0.0))
    weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    density = True
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Explicit numpy array bins with random normals
    x = np.random.randn(30).astype(np.float32)
    y = np.random.randn(30).astype(np.float32)
    bins = (np.arange(-2, 3).astype(np.float32), np.arange(-2, 3).astype(np.float32))
    range_val = ((-2.0, 2.0), (-2.0, 2.0))
    weights = np.random.randn(30).astype(np.float32)
    density = False
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Constant values
    x = np.ones(10, dtype=np.float32) * 5.0
    y = np.ones(10, dtype=np.float32) * 5.0
    bins = (2, 2)
    range_val = ((0.0, 10.0), (0.0, 10.0))
    weights = np.ones(10, dtype=np.float32)
    density = True
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": density,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_9"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_9'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_9'], lib="jax", suffix=9)
