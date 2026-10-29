
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram2d_inputs():
    list_of_inputs = []

    # Case 1: Standard float32 arrays, uniform 2D bin edges, density False
    x = np.array([0.5, 1.2, 2.3, 3.7, 4.1], dtype=np.float32)
    y = np.array([1.1, 2.5, 0.3, 4.8, 3.2], dtype=np.float32)
    bins = np.array(
        [[0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0]], dtype=np.float32
    )
    ranges = np.array([[0.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 2: Float64 arrays with negative values, density True
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float64)
    y = np.array([2.0, -1.0, 0.5, -2.5, 1.0], dtype=np.float64)
    bins = np.array(
        [[-4.0, -2.0, 0.0, 2.0, 4.0], [-3.0, -1.0, 1.0, 3.0, 5.0]],
        dtype=np.float64,
    )
    ranges = np.array([[-4.0, 4.0], [-3.0, 5.0]], dtype=np.float64)
    weights = np.array([0.5, 1.0, 1.5, 2.0, 0.5], dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 3: Integer inputs with bin edges
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    y = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    bins = np.array(
        [[1.0, 3.0, 5.0, 7.0, 9.0], [1.0, 3.0, 5.0, 7.0, 9.0]],
        dtype=np.float32,
    )
    ranges = np.array([[1.0, 9.0], [1.0, 9.0]], dtype=np.float32)
    weights = np.ones(8, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 4: Non-uniformly spaced bin edges
    x = np.array([0.1, 0.5, 1.2, 3.0, 7.5], dtype=np.float32)
    y = np.array([0.2, 0.8, 1.5, 4.0, 8.0], dtype=np.float32)
    bins = np.array(
        [[0.0, 1.0, 5.0, 10.0], [0.0, 2.0, 4.0, 10.0]], dtype=np.float32
    )
    ranges = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 5: Random data, uniform bins
    rng = np.random.RandomState(42)
    x = rng.randn(50).astype(np.float32)
    y = rng.randn(50).astype(np.float32)
    bins = np.array(
        [
            np.linspace(-3.0, 3.0, 6, dtype=np.float32),
            np.linspace(-3.0, 3.0, 6, dtype=np.float32),
        ]
    )
    ranges = np.array([[-3.0, 3.0], [-3.0, 3.0]], dtype=np.float32)
    weights = rng.exponential(1.0, size=50).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 6: Larger sample size with density normalization
    x = rng.uniform(-10.0, 10.0, size=200).astype(np.float64)
    y = rng.uniform(0.0, 20.0, size=200).astype(np.float64)
    bins = np.array(
        [
            np.linspace(-10.0, 10.0, 11, dtype=np.float64),
            np.linspace(0.0, 20.0, 11, dtype=np.float64),
        ]
    )
    ranges = np.array([[-10.0, 10.0], [0.0, 20.0]], dtype=np.float64)
    weights = np.ones(200, dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 7: Data points outside the specified ranges
    x = np.array([-10.0, 0.0, 2.5, 5.0, 20.0], dtype=np.float32)
    y = np.array([-5.0, 1.0, 3.0, 7.0, 15.0], dtype=np.float32)
    bins = np.array(
        [[0.0, 2.0, 4.0, 6.0], [0.0, 2.0, 4.0, 6.0]], dtype=np.float32
    )
    ranges = np.array([[0.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 8: Small 2-element inputs with minimal bins
    x = np.array([0.5, 1.5], dtype=np.float32)
    y = np.array([0.5, 1.5], dtype=np.float32)
    bins = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    ranges = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    weights = np.array([0.2, 0.8], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 9: Integer weights and float64 bins
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    y = np.array([15.0, 25.0, 35.0, 45.0, 55.0], dtype=np.float64)
    bins = np.array(
        [[0.0, 25.0, 50.0, 75.0], [0.0, 25.0, 50.0, 75.0]], dtype=np.float64
    )
    ranges = np.array([[0.0, 75.0], [0.0, 75.0]], dtype=np.float64)
    weights = np.array([2.0, 4.0, 1.0, 3.0, 5.0], dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Case 10: Normal distribution with dense bins and density True
    x = rng.normal(loc=5.0, scale=2.0, size=100).astype(np.float32)
    y = rng.normal(loc=-5.0, scale=1.0, size=100).astype(np.float32)
    bins = np.array(
        [
            np.linspace(0.0, 10.0, 9, dtype=np.float32),
            np.linspace(-10.0, 0.0, 9, dtype=np.float32),
        ]
    )
    ranges = np.array([[0.0, 10.0], [-10.0, 0.0]], dtype=np.float32)
    weights = rng.uniform(0.1, 1.0, size=100).astype(np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x": x,
                "y": y,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

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
