
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays with 10 bins, density=False
    x = np.array([0.1, 0.4, 0.7, 1.2, 1.5, 1.8], dtype=np.float32)
    y = np.array([0.2, 0.3, 0.8, 1.1, 1.4, 1.9], dtype=np.float32)
    bins = np.array(4, dtype=np.int32)
    range_val = [[0.0, 2.0], [0.0, 2.0]]
    weights = np.ones(6, dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 2: Float64 arrays with density=True
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=100).astype(np.float64)
    y = rng.normal(loc=0.0, scale=1.0, size=100).astype(np.float64)
    bins = np.array(10, dtype=np.int64)
    range_val = [[-3.0, 3.0], [-3.0, 3.0]]
    weights = rng.uniform(0.5, 1.5, size=100).astype(np.float64)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 3: Integer inputs with negative and positive ranges
    x = np.array([-10, -5, 0, 5, 10, 15, 20], dtype=np.int32)
    y = np.array([20, 15, 10, 5, 0, -5, -10], dtype=np.int32)
    bins = np.array(5, dtype=np.int32)
    range_val = [[-15.0, 25.0], [-15.0, 25.0]]
    weights = np.array([1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.5], dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 4: Single-element inputs
    x = np.array([2.5], dtype=np.float32)
    y = np.array([3.5], dtype=np.float32)
    bins = np.array(2, dtype=np.int32)
    range_val = [[0.0, 5.0], [0.0, 5.0]]
    weights = np.array([1.0], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 5: Asymmetric range with int64 coordinates
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    y = np.array([-40, -30, -20, -10, 0, 10, 20, 30], dtype=np.int64)
    bins = np.array(8, dtype=np.int32)
    range_val = [[0.0, 100.0], [-50.0, 50.0]]
    weights = np.ones(8, dtype=np.float64)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 6: Large number of points with negative weights
    x = rng.uniform(-10.0, 10.0, size=500).astype(np.float32)
    y = rng.uniform(-5.0, 15.0, size=500).astype(np.float32)
    bins = np.array(20, dtype=np.int32)
    range_val = [[-10.0, 10.0], [-5.0, 15.0]]
    weights = rng.uniform(-1.0, 1.0, size=500).astype(np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 7: Minimum number of bins (1 bin)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    bins = np.array(1, dtype=np.int32)
    range_val = [[0.0, 10.0], [0.0, 10.0]]
    weights = np.array([0.5, 1.5, 2.0], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 8: Uniform distributions with 50 bins
    x = rng.uniform(0.0, 1.0, size=1000).astype(np.float64)
    y = rng.uniform(0.0, 1.0, size=1000).astype(np.float64)
    bins = np.array(50, dtype=np.int64)
    range_val = [[0.0, 1.0], [0.0, 1.0]]
    weights = np.ones(1000, dtype=np.float64)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 9: Non-overlapping data and range
    x = np.array([100.0, 101.0, 102.0], dtype=np.float32)
    y = np.array([200.0, 201.0, 202.0], dtype=np.float32)
    bins = np.array(5, dtype=np.int32)
    range_val = [[0.0, 10.0], [0.0, 10.0]]
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    # Input 10: Tightly clustered data with high resolution bins
    x = rng.normal(loc=5.0, scale=0.1, size=250).astype(np.float32)
    y = rng.normal(loc=-5.0, scale=0.1, size=250).astype(np.float32)
    bins = np.array(25, dtype=np.int32)
    range_val = [[4.5, 5.5], [-5.5, -4.5]]
    weights = rng.exponential(scale=1.0, size=250).astype(np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_val,
            "weights": weights,
            "density": density,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram2d_2"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_2'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_2'], lib="jax", suffix=2)
