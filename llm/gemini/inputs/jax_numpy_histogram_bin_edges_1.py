
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = int(5)
    range_tuple = (0.0, 10.0)
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 2: 2D array with negative numbers
    a = np.array([[-5.0, -2.0, 0.0], [1.0, 3.0, 7.0]], dtype=np.float32)
    bins = int(10)
    range_tuple = (-10.0, 10.0)
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 3: 1D integer array
    a = np.arange(20, dtype=np.int32)
    bins = int(4)
    range_tuple = (0, 20)
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 4: 3D float64 array
    a = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float64)
    bins = int(8)
    range_tuple = (-1.0, 1.0)
    weights = np.random.uniform(0.1, 1.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 5: Single bin
    a = np.array([10.5, 20.3, 15.2, 12.8], dtype=np.float32)
    bins = int(1)
    range_tuple = (10.0, 25.0)
    weights = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 6: Large number of bins
    a = np.linspace(-50.0, 50.0, 100, dtype=np.float32)
    bins = int(50)
    range_tuple = (-60.0, 60.0)
    weights = np.ones(100, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 7: 4D tensor with int64 elements
    a = np.random.randint(-100, 100, size=(2, 2, 3, 3), dtype=np.int64)
    bins = int(20)
    range_tuple = (-100, 100)
    weights = np.ones((2, 2, 3, 3), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 8: Float16 precision
    a = np.array([-2.5, -0.5, 0.5, 2.5], dtype=np.float16)
    bins = int(6)
    range_tuple = (-3.0, 3.0)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float16)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 9: Single element array
    a = np.array([42.0], dtype=np.float32)
    bins = int(2)
    range_tuple = (40.0, 50.0)
    weights = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 10: Non-zero positive integer range
    a = np.random.uniform(100.0, 200.0, size=(30,)).astype(np.float32)
    bins = int(15)
    range_tuple = (100.0, 200.0)
    weights = np.random.uniform(0.0, 1.0, size=(30,)).astype(np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    # Input 11: 1D uint8 array
    a = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
    bins = int(8)
    range_tuple = (0, 256)
    weights = np.ones(6, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "bins": bins,
            "range": range_tuple,
            "weights": weights,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.histogram_bin_edges_1"] = histogram_bin_edges_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_bin_edges_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_bin_edges_1'.")


check_valid('jax.numpy.histogram_bin_edges', generated_inputs['jax.numpy.histogram_bin_edges_1'], lib="jax", suffix=1)
