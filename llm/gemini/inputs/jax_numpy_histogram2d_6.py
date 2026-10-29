
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays, small bin count, density=False
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    bins = 5
    hist_range = ((0.0, 6.0), (0.0, 6.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 2: Float64 arrays, normalized density=True
    x = np.linspace(-3.0, 3.0, 50, dtype=np.float64)
    y = np.linspace(-3.0, 3.0, 50, dtype=np.float64)
    weights = np.random.uniform(0.5, 1.5, size=50).astype(np.float64)
    bins = 10
    hist_range = ((-3.0, 3.0), (-3.0, 3.0))
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 3: Integer inputs with float32 weights
    x = np.array([-10, -5, 0, 5, 10, 15], dtype=np.int32)
    y = np.array([10, 5, 0, -5, -10, -15], dtype=np.int32)
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    bins = 4
    hist_range = ((-15.0, 20.0), (-20.0, 15.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 4: Moderate size random arrays, density=True
    np.random.seed(42)
    x = np.random.normal(loc=0.0, scale=1.0, size=100).astype(np.float32)
    y = np.random.normal(loc=1.0, scale=2.0, size=100).astype(np.float32)
    weights = np.random.exponential(scale=1.0, size=100).astype(np.float32)
    bins = 8
    hist_range = ((-4.0, 4.0), (-5.0, 7.0))
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 5: Negative values range, density=False
    x = np.array([-50.0, -40.0, -30.0, -20.0, -10.0], dtype=np.float32)
    y = np.array([-25.0, -20.0, -15.0, -10.0, -5.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
    bins = 3
    hist_range = ((-60.0, 0.0), (-30.0, 0.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 6: Large number of bins, float64
    x = np.random.uniform(0.0, 1.0, size=200).astype(np.float64)
    y = np.random.uniform(0.0, 1.0, size=200).astype(np.float64)
    weights = np.ones(200, dtype=np.float64)
    bins = 20
    hist_range = ((0.0, 1.0), (0.0, 1.0))
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 7: Int64 arrays with float64 weights
    x = np.arange(20, dtype=np.int64)
    y = np.arange(20, dtype=np.int64) * 2
    weights = np.linspace(0.1, 2.0, 20, dtype=np.float64)
    bins = 5
    hist_range = ((0.0, 25.0), (0.0, 50.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 8: Minimal elements with bins=2
    x = np.array([0.5, 1.5], dtype=np.float32)
    y = np.array([1.5, 0.5], dtype=np.float32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    bins = 2
    hist_range = ((0.0, 2.0), (0.0, 2.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 9: Asymmetric range with density=True
    x = np.linspace(10.0, 100.0, 40, dtype=np.float32)
    y = np.linspace(-100.0, -10.0, 40, dtype=np.float32)
    weights = np.full(40, 0.5, dtype=np.float32)
    bins = 6
    hist_range = ((0.0, 120.0), (-120.0, 0.0))
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

    # Input 10: Float64 high precision data with negative and fractional values
    x = np.array([-2.5, -1.25, 0.0, 1.25, 2.5], dtype=np.float64)
    y = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float64)
    weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    bins = 5
    hist_range = ((-3.0, 3.0), (-1.0, 1.0))
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": hist_range,
            "weights": weights,
            "density": density,
        }
    )

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
