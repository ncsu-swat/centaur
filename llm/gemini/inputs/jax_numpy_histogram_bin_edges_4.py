
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 input with matching weights
    a = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    bins = 5
    range_val = np.array([1.0, 6.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 2: Integer input with negative values
    a = np.array([-10, 0, 10, 20, 30], dtype=np.int32)
    bins = 4
    range_val = np.array([-15, 35], dtype=np.int32)
    weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 3: 2D array input
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bins = 3
    range_val = np.array([0.0, 5.0], dtype=np.float32)
    weights = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 4: 3D random float array
    a = np.random.randn(2, 3, 4).astype(np.float32)
    bins = 10
    range_val = np.array([-3.0, 3.0], dtype=np.float32)
    weights = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 5: Float64 high precision
    a = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    bins = 2
    range_val = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 6: Large values range
    a = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
    bins = 3
    range_val = np.array([500.0, 3500.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 7: Single bin edge test
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = 1
    range_val = np.array([1.0, 5.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 8: Exclusively negative elements and range
    a = np.array([-5.0, -4.0, -3.0], dtype=np.float32)
    bins = 2
    range_val = np.array([-6.0, -2.0], dtype=np.float32)
    weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 9: Large number of bins with randomly generated values
    a = np.random.rand(100).astype(np.float32)
    bins = 50
    range_val = np.array([0.0, 1.0], dtype=np.float32)
    weights = np.ones(100, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 10: Constant value array with a defined range
    a = np.zeros(10, dtype=np.float32)
    bins = 5
    range_val = np.array([0.0, 1.0], dtype=np.float32)
    weights = np.ones(10, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    return list_of_inputs

generated_inputs["jax.numpy.histogram_bin_edges_4"] = histogram_bin_edges_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_bin_edges_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_bin_edges_4'.")


check_valid('jax.numpy.histogram_bin_edges', generated_inputs['jax.numpy.histogram_bin_edges_4'], lib="jax", suffix=4)
