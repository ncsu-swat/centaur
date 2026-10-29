
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 1D sample, float32, density=False
    sample = np.random.randn(100, 1).astype(np.float32)
    bins = [10]
    ranges = [[-3.0, 3.0]]
    weights = np.random.uniform(0.1, 1.0, size=(100,)).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: 2D sample, float64, density=True
    sample = np.random.uniform(0.0, 10.0, size=(150, 2)).astype(np.float64)
    bins = [5, 8]
    ranges = [[0.0, 10.0], [0.0, 10.0]]
    weights = np.ones(150, dtype=np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: 3D sample, float32, density=False
    sample = np.random.randn(200, 3).astype(np.float32)
    bins = [4, 4, 4]
    ranges = [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
    weights = np.random.exponential(scale=1.0, size=(200,)).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: 4D sample, float32, density=True
    sample = np.random.uniform(-5.0, 5.0, size=(80, 4)).astype(np.float32)
    bins = [2, 3, 2, 3]
    ranges = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    weights = np.random.uniform(0.5, 2.0, size=(80,)).astype(np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: 2D sample with negative values only
    sample = np.random.uniform(-20.0, -1.0, size=(120, 2)).astype(np.float32)
    bins = [6, 7]
    ranges = [[-20.0, -1.0], [-20.0, -1.0]]
    weights = np.random.uniform(0.1, 1.0, size=(120,)).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: 5D sample, float32, small bin counts
    sample = np.random.randn(50, 5).astype(np.float32)
    bins = [2, 2, 2, 2, 2]
    ranges = [[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]
    weights = np.ones(50, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: Single-point sample (edge case N=1), 2D
    sample = np.array([[0.5, 0.5]], dtype=np.float32)
    bins = [4, 4]
    ranges = [[0.0, 1.0], [0.0, 1.0]]
    weights = np.array([1.0], dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: 3D sample with asymmetric bin counts and ranges
    sample = np.random.uniform(-10.0, 10.0, size=(300, 3)).astype(np.float64)
    bins = [10, 5, 2]
    ranges = [[-10.0, 10.0], [-5.0, 5.0], [-2.0, 2.0]]
    weights = np.random.uniform(0.0, 5.0, size=(300,)).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: 2D sample with large number of points
    sample = np.random.randn(1000, 2).astype(np.float32)
    bins = [12, 12]
    ranges = [[-4.0, 4.0], [-4.0, 4.0]]
    weights = np.random.uniform(0.1, 0.9, size=(1000,)).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: 1D sample, float64, density=True
    sample = np.random.normal(loc=5.0, scale=2.0, size=(250, 1)).astype(
        np.float64
    )
    bins = [15]
    ranges = [[0.0, 10.0]]
    weights = np.random.uniform(1.0, 2.0, size=(250,)).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": ranges,
                "weights": weights,
                "density": density,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogramdd_3"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_3'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_3'], lib="jax", suffix=3)
