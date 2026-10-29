
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D inputs with integer bin tuple
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    bins = (5, 5)
    range_val = ((0.0, 10.0), (0.0, 10.0))
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: Negative and positive float64 values, density=True
    x = np.array([-2.5, -1.0, 0.0, 1.5, 3.2], dtype=np.float64)
    y = np.array([-3.0, 0.5, 1.2, -0.8, 2.0], dtype=np.float64)
    bins = (4, 6)
    range_val = ((-4.0, 4.0), (-4.0, 4.0))
    weights = np.array([0.5, 1.5, 2.0, 0.1, 1.0], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: Int32 inputs with int bin counts and density=True
    x = np.array([1, 5, 10, 15, 20, 25], dtype=np.int32)
    y = np.array([20, 15, 10, 5, 1, 0], dtype=np.int32)
    bins = (3, 3)
    range_val = ((0.0, 30.0), (0.0, 30.0))
    weights = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: Large random sample with float32
    np.random.seed(42)
    x = np.random.uniform(-10.0, 10.0, size=(100,)).astype(np.float32)
    y = np.random.uniform(-5.0, 5.0, size=(100,)).astype(np.float32)
    bins = (10, 8)
    range_val = ((-10.0, 10.0), (-5.0, 5.0))
    weights = np.random.uniform(0.1, 1.0, size=(100,)).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: Single-element arrays
    x = np.array([2.5], dtype=np.float32)
    y = np.array([3.5], dtype=np.float32)
    bins = (2, 2)
    range_val = ((0.0, 5.0), (0.0, 5.0))
    weights = np.array([1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: Int64 input arrays
    x = np.array([-50, -20, 0, 20, 50], dtype=np.int64)
    y = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    bins = (5, 4)
    range_val = ((-60.0, 60.0), (0.0, 60.0))
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Asymmetric bin counts with float32
    x = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    y = np.linspace(10.0, 20.0, 20, dtype=np.float32)
    bins = (2, 10)
    range_val = ((0.0, 1.0), (10.0, 20.0))
    weights = np.ones(20, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: Float64 inputs with varied weights
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    y = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    bins = (4, 4)
    range_val = ((0.0, 1.0), (0.0, 1.0))
    weights = np.array(
        [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1], dtype=np.float64
    )
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: Negative float ranges and bins
    x = np.array([-10.0, -8.0, -6.0, -4.0, -2.0], dtype=np.float32)
    y = np.array([-5.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float32)
    bins = (3, 5)
    range_val = ((-12.0, 0.0), (-6.0, 0.0))
    weights = np.array([1.5, 0.5, 2.5, 1.0, 0.5], dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: Larger number of bins
    x = np.linspace(-5.0, 5.0, 50, dtype=np.float32)
    y = np.linspace(-5.0, 5.0, 50, dtype=np.float32)
    bins = (20, 20)
    range_val = ((-5.0, 5.0), (-5.0, 5.0))
    weights = np.ones(50, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_val,
                "weights": weights,
                "density": density,
            }
        )
    )

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
