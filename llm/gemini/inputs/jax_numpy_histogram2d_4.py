
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 inputs with uniform range and equal bins
    x = np.array([0.1, 0.4, 0.7, 0.2, 0.9], dtype=np.float32)
    y = np.array([0.3, 0.5, 0.1, 0.8, 0.2], dtype=np.float32)
    bins = (5, 5)
    range_param = [[0.0, 1.0], [0.0, 1.0]]
    weights = np.ones(5, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: Different number of bins per dimension and density=True
    x = np.array([-0.5, 0.0, 0.5, 0.8, -0.2], dtype=np.float32)
    y = np.array([-1.5, -0.5, 0.5, 1.0, 1.5], dtype=np.float32)
    bins = (10, 8)
    range_param = [[-1.0, 1.0], [-2.0, 2.0]]
    weights = np.array([0.5, 1.0, 1.5, 2.0, 0.2], dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: Integer data inputs
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    y = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    bins = (4, 4)
    range_param = [[0.0, 10.0], [0.0, 10.0]]
    weights = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: Float64 inputs with negative and positive values
    np.random.seed(42)
    x = np.random.randn(50).astype(np.float64)
    y = np.random.randn(50).astype(np.float64)
    bins = (3, 7)
    range_param = [[-3.0, 3.0], [-3.0, 3.0]]
    weights = np.random.uniform(0.1, 1.0, size=50).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: Larger dataset with 500 points
    x = np.random.uniform(0.0, 100.0, size=500).astype(np.float32)
    y = np.random.uniform(0.0, 100.0, size=500).astype(np.float32)
    bins = (20, 20)
    range_param = [[0.0, 100.0], [0.0, 100.0]]
    weights = np.ones(500, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: Minimal size dataset (2 points)
    x = np.array([-0.5, 0.5], dtype=np.float32)
    y = np.array([0.2, -0.2], dtype=np.float32)
    bins = (2, 2)
    range_param = [[-1.0, 1.0], [-1.0, 1.0]]
    weights = np.array([1.0, 2.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Explicit bin edges supplied as a tuple of arrays
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = (
        np.array([1.0, 3.0, 5.0], dtype=np.float32),
        np.array([1.0, 3.0, 6.0], dtype=np.float32),
    )
    range_param = [[1.0, 5.0], [1.0, 6.0]]
    weights = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: Asymmetric bins with density=True
    x = np.linspace(-5.0, 5.0, 100, dtype=np.float32)
    y = np.linspace(0.0, 10.0, 100, dtype=np.float32)
    bins = (15, 3)
    range_param = [[-5.0, 5.0], [0.0, 10.0]]
    weights = np.linspace(0.1, 1.0, 100, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: Non-square range wider than data range
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
    bins = (8, 12)
    range_param = [[-50.0, 50.0], [-20.0, 20.0]]
    weights = np.array([2.5, 1.0, 0.5], dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: Constant values
    x = np.zeros(10, dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    bins = (5, 5)
    range_param = [[-1.0, 1.0], [-1.0, 1.0]]
    weights = np.ones(10, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

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
