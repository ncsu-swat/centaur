
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays with default-like settings
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 15.0, 19.0, 25.0], dtype=np.float32)
    y = np.array([2.0, 5.0, 6.0, 8.0, 13.0, 16.0, 17.0, 18.0], dtype=np.float32)
    weights = np.ones(8, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 8,
        "range": [[0.0, 30.0], [0.0, 30.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64 arrays with density=True
    x = np.linspace(-5.0, 5.0, 50, dtype=np.float64)
    y = np.linspace(-10.0, 10.0, 50, dtype=np.float64)
    weights = np.random.uniform(0.5, 1.5, size=50).astype(np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 10,
        "range": [[-5.0, 5.0], [-10.0, 10.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Integer inputs with small number of bins
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int32)
    weights = np.ones(10, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 3,
        "range": [[0.0, 10.0], [0.0, 10.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative ranges and data
    x = np.array([-15.0, -10.0, -5.0, -1.0, -0.5], dtype=np.float32)
    y = np.array([-20.0, -12.0, -8.0, -4.0, -1.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 4,
        "range": [[-25.0, 0.0], [-25.0, 0.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger dataset (1000 points) with higher bin count
    rng = np.random.RandomState(42)
    x = rng.randn(1000).astype(np.float32)
    y = rng.randn(1000).astype(np.float32)
    weights = rng.exponential(scale=1.0, size=1000).astype(np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 25,
        "range": [[-3.0, 3.0], [-3.0, 3.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Minimal dataset (2 points) with 2 bins
    x = np.array([1.5, 3.5], dtype=np.float32)
    y = np.array([2.5, 4.5], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 2,
        "range": [[0.0, 5.0], [0.0, 5.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 precision with density=True
    x = np.array([-100.0, 0.0, 100.0], dtype=np.float64)
    y = np.array([-50.0, 0.0, 50.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 5,
        "range": [[-150.0, 150.0], [-100.0, 100.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Range narrower than data bounds (clipping values)
    x = np.linspace(0.0, 100.0, 20, dtype=np.float32)
    y = np.linspace(0.0, 100.0, 20, dtype=np.float32)
    weights = np.full(20, 2.0, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 5,
        "range": [[20.0, 80.0], [20.0, 80.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Asymmetric range with density=True
    x = np.arange(16, dtype=np.float32)
    y = np.arange(16, dtype=np.float32) ** 1.5
    weights = np.ones(16, dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 6,
        "range": [[0.0, 20.0], [0.0, 70.0]],
        "weights": weights,
        "density": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single bin configuration
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    y = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "y": y,
        "bins": 1,
        "range": [[0.0, 1.0], [0.0, 1.0]],
        "weights": weights,
        "density": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
