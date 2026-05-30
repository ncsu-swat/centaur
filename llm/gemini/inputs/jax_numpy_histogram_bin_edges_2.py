
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D array with 5 bins
    a = np.array([2, 5, 3, 6, 4, 1], dtype=np.float32)
    bins = np.array(5, dtype=np.int32)
    range_val = (1.0, 6.0)
    weights = np.ones_like(a)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 2: Random standard normal array with 10 bins
    a = np.random.randn(100).astype(np.float32)
    bins = np.array(10, dtype=np.int32)
    range_val = (-3.0, 3.0)
    weights = np.random.rand(100).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 3: Float64 array with 20 bins and custom range
    a = np.random.rand(50).astype(np.float64)
    bins = np.array(20, dtype=np.int32)
    range_val = (0.0, 1.0)
    weights = np.random.rand(50).astype(np.float64)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 4: Pre-specified bin edges as tensor
    a = np.arange(10, dtype=np.int32)
    bins = np.array([0, 2, 5, 10], dtype=np.float32)
    range_val = (0.0, 10.0)
    weights = np.ones(10, dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 5: 2D array of inputs
    a = np.random.randn(10, 10).astype(np.float32)
    bins = np.array(15, dtype=np.int32)
    range_val = (-2.0, 2.0)
    weights = np.ones((10, 10), dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 6: Symmetric negative/positive values
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    bins = np.array(4, dtype=np.int32)
    range_val = (-10.0, 10.0)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 7: Uniformly distributed data with larger bin count
    a = np.random.uniform(-100, 100, size=(50,)).astype(np.float32)
    bins = np.array(50, dtype=np.int32)
    range_val = (-100.0, 100.0)
    weights = np.random.rand(50).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 8: Int64 inputs and bins
    a = np.array([1, 2, 3], dtype=np.int64)
    bins = np.array(2, dtype=np.int64)
    range_val = (1.0, 3.0)
    weights = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 9: 3D tensor input
    a = np.random.randn(5, 5, 5).astype(np.float32)
    bins = np.array(8, dtype=np.int32)
    range_val = (-4.0, 4.0)
    weights = np.random.rand(5, 5, 5).astype(np.float32)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    # Input 10: High-precision linspace input
    a = np.linspace(-5, 5, 100).astype(np.float64)
    bins = np.array(12, dtype=np.int32)
    range_val = (-5.0, 5.0)
    weights = np.linspace(0.1, 1.0, 100).astype(np.float64)
    list_of_inputs.append({
        "a": a,
        "bins": bins,
        "range": range_val,
        "weights": weights
    })

    return list_of_inputs

generated_inputs["jax.numpy.histogram_bin_edges_2"] = histogram_bin_edges_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_bin_edges_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_bin_edges_2'.")


check_valid('jax.numpy.histogram_bin_edges', generated_inputs['jax.numpy.histogram_bin_edges_2'], lib="jax", suffix=2)
