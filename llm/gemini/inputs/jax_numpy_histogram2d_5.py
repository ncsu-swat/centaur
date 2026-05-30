
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 inputs
    x = np.random.uniform(0, 10, 100).astype(np.float32)
    y = np.random.uniform(0, 10, 100).astype(np.float32)
    bins = np.array([np.linspace(0, 10, 5), np.linspace(0, 10, 5)]).astype(np.float32)
    range_val = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    weights = np.ones(100).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values
    x = np.random.uniform(-5, 5, 50).astype(np.float32)
    y = np.random.uniform(-5, 5, 50).astype(np.float32)
    bins = np.array([np.linspace(-5, 5, 6), np.linspace(-5, 5, 6)]).astype(np.float32)
    range_val = np.array([[-5.0, 5.0], [-5.0, 5.0]], dtype=np.float32)
    weights = np.random.uniform(0, 1, 50).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 high precision
    x = np.random.uniform(100, 200, 150).astype(np.float64)
    y = np.random.uniform(100, 200, 150).astype(np.float64)
    bins = np.array([np.linspace(100, 200, 11), np.linspace(100, 200, 11)]).astype(np.float64)
    range_val = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float64)
    weights = np.random.uniform(1, 2, 150).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer inputs
    x = np.random.randint(-10, 10, 30).astype(np.int32)
    y = np.random.randint(-10, 10, 30).astype(np.int32)
    bins = np.array([[-10, -5, 0, 5, 10], [-10, -5, 0, 5, 10]]).astype(np.int32)
    range_val = np.array([[-10, 10], [-10, 10]], dtype=np.int32)
    weights = np.ones(30).astype(np.int32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Unequal range but same shape bin edges in X and Y dimensions
    x = np.random.uniform(0, 10, 80).astype(np.float32)
    y = np.random.uniform(-5, 5, 80).astype(np.float32)
    bins = np.array([[0., 2., 5., 10.], [-5., -2., 2., 5.]]).astype(np.float32)
    range_val = np.array([[0., 10.], [-5., 5.]], dtype=np.float32)
    weights = np.random.uniform(0.1, 1.0, 80).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large inputs
    x = np.random.normal(0, 1, 1000).astype(np.float32)
    y = np.random.normal(0, 1, 1000).astype(np.float32)
    bins = np.array([np.linspace(-3, 3, 21), np.linspace(-3, 3, 21)]).astype(np.float32)
    range_val = np.array([[-3.0, 3.0], [-3.0, 3.0]], dtype=np.float32)
    weights = np.ones(1000).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Small size inputs
    x = np.array([1.5, 2.5, 3.5]).astype(np.float32)
    y = np.array([4.5, 5.5, 6.5]).astype(np.float32)
    bins = np.array([[1., 2., 3., 4.], [4., 5., 6., 7.]]).astype(np.float32)
    range_val = np.array([[1.0, 4.0], [4.0, 7.0]], dtype=np.float32)
    weights = np.array([0.5, 0.5, 0.5]).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative weights
    x = np.random.uniform(0, 1, 40).astype(np.float32)
    y = np.random.uniform(0, 1, 40).astype(np.float32)
    bins = np.array([np.linspace(0, 1, 5), np.linspace(0, 1, 5)]).astype(np.float32)
    range_val = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    weights = np.random.uniform(-1, 1, 40).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element arrays
    x = np.array([5.0]).astype(np.float32)
    y = np.array([5.0]).astype(np.float32)
    bins = np.array([[0.0, 10.0], [0.0, 10.0]]).astype(np.float32)
    range_val = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    weights = np.array([1.0]).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int16 inputs
    x = np.random.randint(0, 50, 100).astype(np.int16)
    y = np.random.randint(0, 50, 100).astype(np.int16)
    bins = np.array([np.arange(0, 60, 10), np.arange(0, 60, 10)]).astype(np.int16)
    range_val = np.array([[0, 50], [0, 50]], dtype=np.int16)
    weights = np.ones(100).astype(np.int16)
    input_dict = {
        "x": x,
        "y": y,
        "bins": bins,
        "range": range_val,
        "weights": weights,
        "density": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram2d_5"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_5'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_5'], lib="jax", suffix=5)
