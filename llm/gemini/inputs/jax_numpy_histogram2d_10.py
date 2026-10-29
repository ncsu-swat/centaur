
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays, uniform binning, density=False
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    bins = np.array(
        [[0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 5.0, 7.0]], dtype=np.float32
    )
    rng = ((0.0, 6.0), (1.0, 7.0))
    weights = np.ones(5, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: Float64 arrays with negative values, density=True
    x = np.array([-3.5, -1.2, 0.0, 1.5, 4.2], dtype=np.float64)
    y = np.array([-2.1, -0.5, 1.1, 2.3, 3.8], dtype=np.float64)
    bins = np.array(
        [[-5.0, -2.0, 1.0, 5.0], [-3.0, 0.0, 2.0, 5.0]], dtype=np.float64
    )
    rng = ((-5.0, 5.0), (-3.0, 5.0))
    weights = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: Int32 input arrays, non-uniform bins, density=False
    x = np.array([0, 10, 20, 30, 40, 50], dtype=np.int32)
    y = np.array([5, 15, 25, 35, 45, 55], dtype=np.int32)
    bins = np.array(
        [[0.0, 15.0, 35.0, 60.0], [0.0, 20.0, 40.0, 70.0]], dtype=np.float32
    )
    rng = ((0.0, 60.0), (0.0, 70.0))
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: Large random sample, density=True
    np.random.seed(42)
    x = np.random.randn(200).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    bins = np.stack(
        [
            np.linspace(-3.0, 3.0, 11, dtype=np.float32),
            np.linspace(-3.0, 3.0, 11, dtype=np.float32),
        ]
    )
    rng = ((-3.0, 3.0), (-3.0, 3.0))
    weights = np.random.uniform(0.1, 1.0, size=200).astype(np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: Single bin in each dimension (minimal edges), density=False
    x = np.array([0.5, 0.2, 0.8], dtype=np.float32)
    y = np.array([0.1, 0.9, 0.4], dtype=np.float32)
    bins = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    rng = ((0.0, 1.0), (0.0, 1.0))
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: Symmetric grid binning, density=True
    x = np.linspace(-10.0, 10.0, 50, dtype=np.float64)
    y = np.linspace(0.0, 100.0, 50, dtype=np.float64)
    bins = np.stack(
        [
            np.linspace(-10.0, 10.0, 6, dtype=np.float64),
            np.linspace(0.0, 100.0, 6, dtype=np.float64),
        ]
    )
    rng = ((-10.0, 10.0), (0.0, 100.0))
    weights = np.full(50, 0.02, dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Int64 arrays and weights, density=False
    x = np.array([100, 200, 300, 400], dtype=np.int64)
    y = np.array([500, 600, 700, 800], dtype=np.int64)
    bins = np.array([[50, 250, 450], [450, 650, 850]], dtype=np.float64)
    rng = ((50.0, 450.0), (450.0, 850.0))
    weights = np.array([10, 20, 30, 40], dtype=np.int64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: Data with negative weights, density=False
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    y = np.array([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32)
    bins = np.array(
        [[-3.0, -1.0, 1.0, 3.0], [-5.0, -1.0, 2.0, 5.0]], dtype=np.float32
    )
    rng = ((-3.0, 3.0), (-5.0, 5.0))
    weights = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: Fine grid binning with 100 points, density=True
    np.random.seed(123)
    x = np.random.exponential(scale=2.0, size=100).astype(np.float32)
    y = np.random.exponential(scale=3.0, size=100).astype(np.float32)
    bins = np.stack(
        [
            np.linspace(0.0, 10.0, 21, dtype=np.float32),
            np.linspace(0.0, 15.0, 21, dtype=np.float32),
        ]
    )
    rng = ((0.0, 10.0), (0.0, 15.0))
    weights = np.ones(100, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: Repeated/duplicate values spanning exact bin boundaries, density=False
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    bins = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    rng = ((0.0, 2.0), (0.0, 2.0))
    weights = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": rng,
                "weights": weights,
                "density": density,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram2d_10"] = histogram2d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram2d_10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram2d_10'.")


check_valid('jax.numpy.histogram2d', generated_inputs['jax.numpy.histogram2d_10'], lib="jax", suffix=10)
