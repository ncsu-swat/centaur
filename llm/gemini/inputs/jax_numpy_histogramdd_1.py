
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogramdd_inputs():
    list_of_inputs = []

    # Input 1: 2D sample, float32, basic parameters
    sample = np.random.randn(50, 2).astype(np.float32)
    weights = np.ones(50, dtype=np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 5,
            "range": [(-3.0, 3.0), (-3.0, 3.0)],
            "weights": weights,
            "density": False,
        }
    )

    # Input 2: 1D sample with shape (N, 1), density=True
    sample = np.linspace(-2.0, 2.0, 30, dtype=np.float64).reshape(-1, 1)
    weights = np.random.uniform(0.5, 1.5, size=30).astype(np.float64)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 10,
            "range": [(-2.5, 2.5)],
            "weights": weights,
            "density": True,
        }
    )

    # Input 3: 3D sample, float64, non-negative range
    sample = np.random.uniform(0.0, 10.0, size=(100, 3)).astype(np.float64)
    weights = np.random.uniform(0.1, 2.0, size=100).astype(np.float64)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 4,
            "range": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            "weights": weights,
            "density": False,
        }
    )

    # Input 4: 4D sample, float32, density=True
    sample = np.random.normal(loc=0.0, scale=1.0, size=(80, 4)).astype(
        np.float32
    )
    weights = np.random.exponential(scale=1.0, size=80).astype(np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 3,
            "range": [(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)],
            "weights": weights,
            "density": True,
        }
    )

    # Input 5: 2D integer sample
    sample = np.random.randint(-10, 10, size=(60, 2)).astype(np.int32)
    weights = np.ones(60, dtype=np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 6,
            "range": [(-10.0, 10.0), (-10.0, 10.0)],
            "weights": weights,
            "density": False,
        }
    )

    # Input 6: 2D sample with negative range and asymmetric bounds
    sample = np.random.uniform(-5.0, -1.0, size=(40, 2)).astype(np.float32)
    weights = np.random.uniform(1.0, 5.0, size=40).astype(np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 8,
            "range": [(-6.0, 0.0), (-5.0, -1.0)],
            "weights": weights,
            "density": True,
        }
    )

    # Input 7: 5D sample, small bin count
    sample = np.random.randn(120, 5).astype(np.float32)
    weights = np.abs(np.random.randn(120)).astype(np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 2,
            "range": [
                (-3.0, 3.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ],
            "weights": weights,
            "density": False,
        }
    )

    # Input 8: 3D sample with integer weights
    sample = np.random.uniform(-1.0, 1.0, size=(75, 3)).astype(np.float64)
    weights = np.random.randint(1, 5, size=75).astype(np.int64)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 7,
            "range": [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)],
            "weights": weights,
            "density": True,
        }
    )

    # Input 9: Single-point 2D sample
    sample = np.array([[0.5, -0.5]], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 3,
            "range": [(0.0, 1.0), (-1.0, 0.0)],
            "weights": weights,
            "density": False,
        }
    )

    # Input 10: 2D sample with large number of points and float64 weights
    sample = np.random.randn(200, 2).astype(np.float64)
    weights = np.random.uniform(0.0, 1.0, size=200).astype(np.float64)
    list_of_inputs.append(
        {
            "sample": sample,
            "bins": 12,
            "range": [(-4.0, 4.0), (-4.0, 4.0)],
            "weights": weights,
            "density": True,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogramdd_1"] = histogramdd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogramdd_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogramdd_1'.")


check_valid('jax.numpy.histogramdd', generated_inputs['jax.numpy.histogramdd_1'], lib="jax", suffix=1)
