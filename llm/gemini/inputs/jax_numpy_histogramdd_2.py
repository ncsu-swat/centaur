
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 standard normal data
    np.random.seed(42)
    sample = np.random.randn(100, 2).astype(np.float32)
    bins = np.array([10, 10], dtype=np.int32)
    range_param = [(-3.0, 3.0), (-3.0, 3.0)]
    weights = np.ones(100, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 2: 3D float64 uniform data with density=True
    sample = np.random.uniform(0.0, 10.0, size=(50, 3)).astype(np.float64)
    bins = np.array([5, 8, 6], dtype=np.int64)
    range_param = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    weights = np.random.uniform(0.1, 1.0, size=50).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 3: 1D data represented as (200, 1)
    sample = np.random.randn(200, 1).astype(np.float32)
    bins = np.array([20], dtype=np.int32)
    range_param = [(-10.0, 10.0)]
    weights = np.random.randn(200).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 4: 4D data with different ranges and density=True
    sample = np.random.randn(80, 4).astype(np.float32)
    bins = np.array([4, 4, 4, 4], dtype=np.int32)
    range_param = [(-1.0, 1.0), (-2.0, 2.0), (0.0, 5.0), (-3.0, 0.0)]
    weights = np.abs(np.random.randn(80)).astype(np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 5: 2D integer sample
    sample = np.random.randint(0, 50, size=(120, 2)).astype(np.int32)
    bins = np.array([6, 12], dtype=np.int32)
    range_param = [(0.0, 50.0), (0.0, 50.0)]
    weights = np.ones(120, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 6: 5D high-dimensional data with asymmetric bin counts
    sample = np.random.uniform(-2.0, 2.0, size=(40, 5)).astype(np.float64)
    bins = np.array([2, 3, 2, 4, 3], dtype=np.int64)
    range_param = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    weights = np.random.uniform(1.0, 2.0, size=40).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 7: 2D float64 with linearly spaced weights
    sample = np.random.randn(300, 2).astype(np.float64)
    bins = np.array([15, 25], dtype=np.int32)
    range_param = [(-5.0, 5.0), (-10.0, 10.0)]
    weights = np.linspace(0.1, 1.0, 300, dtype=np.float64)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 8: 1D feature space (shape 150, 1)
    sample = np.random.uniform(-3.0, 3.0, size=(150, 1)).astype(np.float32)
    bins = np.array([15], dtype=np.int32)
    range_param = [(-4.0, 4.0)]
    weights = np.ones(150, dtype=np.float32)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 9: 3D data with exponential weights
    sample = np.random.randn(75, 3).astype(np.float32)
    bins = np.array([8, 8, 8], dtype=np.int32)
    range_param = [(-2.5, 2.5), (-1.5, 1.5), (-0.5, 3.5)]
    weights = np.random.exponential(1.0, size=75).astype(np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 10: 2D unit square float64 data with density=True
    sample = np.random.rand(64, 2).astype(np.float64)
    bins = np.array([4, 8], dtype=np.int64)
    range_param = [(0.0, 1.0), (0.0, 1.0)]
    weights = np.random.rand(64).astype(np.float64)
    density = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    # Input 11: Small 3D sample with coarse bins and wide range
    sample = np.zeros((10, 3), dtype=np.float32)
    bins = np.array([2, 2, 2], dtype=np.int32)
    range_param = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    weights = np.full(10, 0.5, dtype=np.float32)
    density = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "sample": sample,
                "bins": bins,
                "range": range_param,
                "weights": weights,
                "density": density,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogramdd_2"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_2'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_2'], lib="jax", suffix=2)
