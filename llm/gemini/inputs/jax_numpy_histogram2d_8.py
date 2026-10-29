
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float arrays with scalar integers in bins list
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    bins = [5, 5]
    range_param = ((0.0, 10.0), (0.0, 10.0))
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 2: Negative values, density=True
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    y = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    bins = [4, 4]
    range_param = ((-6.0, 6.0), (-12.0, 12.0))
    weights = np.array([0.5, 1.5, 2.0, 1.0, 0.5], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 3: Uniform integer bin counts
    x = np.array([0.1, 1.2, 2.3, 3.4, 4.5, 5.6], dtype=np.float64)
    y = np.array([5.5, 4.4, 3.3, 2.2, 1.1, 0.0], dtype=np.float64)
    bins = [3, 3]
    range_param = ((0.0, 6.0), (0.0, 6.0))
    weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 4: Int32 data types with asymmetric bin counts
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
    y = np.array([5, 15, 25, 35, 45, 55, 65, 75], dtype=np.int32)
    bins = [8, 4]
    range_param = ((0.0, 100.0), (0.0, 100.0))
    weights = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 5: Random normal data
    np.random.seed(42)
    x = np.random.randn(100).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    bins = [10, 10]
    range_param = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.ones(100, dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 6: Asymmetric bins with density
    x = np.array([0.5, 1.5, 3.0, 7.0, 9.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 4.0, 8.0, 10.0], dtype=np.float32)
    bins = [2, 4]
    range_param = ((0.0, 10.0), (0.0, 12.0))
    weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 7: Float64 high precision
    x = np.linspace(-1.0, 1.0, 50, dtype=np.float64)
    y = np.linspace(0.0, 2.0, 50, dtype=np.float64)
    bins = [5, 10]
    range_param = ((-1.0, 1.0), (0.0, 2.0))
    weights = np.linspace(0.1, 1.0, 50, dtype=np.float64)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 8: Single element data
    x = np.array([0.5], dtype=np.float32)
    y = np.array([0.5], dtype=np.float32)
    bins = [2, 2]
    range_param = ((0.0, 1.0), (0.0, 1.0))
    weights = np.array([1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 9: Negative range boundaries
    x = np.array([-10.0, -8.0, -6.0, -4.0, -2.0], dtype=np.float32)
    y = np.array([-20.0, -16.0, -12.0, -8.0, -4.0], dtype=np.float32)
    bins = [3, 3]
    range_param = ((-15.0, 0.0), (-25.0, 0.0))
    weights = np.array([1.0, 1.0, 2.0, 2.0, 1.0], dtype=np.float32)
    density = True
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    # Input 10: Non-square bin counts with float weights
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    y = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    bins = [3, 2]
    range_param = ((0.0, 12.0), (0.0, 10.0))
    weights = np.array(
        [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype=np.float64
    )
    density = False
    list_of_inputs.append(
        {
            "x": x,
            "y": y,
            "bins": bins,
            "range": range_param,
            "weights": weights,
            "density": density,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram2d_8"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_8'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_8'], lib="jax", suffix=8)
