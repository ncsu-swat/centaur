
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = 5
    range_val = [0.0, 6.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 2: Negative values
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    bins = 10
    range_val = [-15.0, 15.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 3: Int32 array, 2D
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    bins = 3
    range_val = [0, 5]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 4: Float64 array, 1D
    a = np.random.uniform(-1.0, 1.0, size=(100,)).astype(np.float64)
    bins = 20
    range_val = [-2.0, 2.0]
    weights = np.random.uniform(0.1, 1.0, size=(100,)).astype(np.float64)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 5: 3D array, JAX histogram flattens it
    a = np.random.randn(2, 3, 4).astype(np.float32)
    bins = 8
    range_val = [-3.0, 3.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 6: Zero range variation
    a = np.zeros(10, dtype=np.float32)
    bins = 5
    range_val = [0.0, 1.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 7: Small bins size
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    bins = 2
    range_val = [10.0, 30.0]
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 8: Uniform integers
    a = np.arange(10, dtype=np.int64)
    bins = 4
    range_val = [0, 10]
    weights = np.ones_like(a, dtype=np.int64)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 9: Large number of bins
    a = np.random.exponential(scale=1.0, size=(50,)).astype(np.float32)
    bins = 100
    range_val = [0.0, 10.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    # Input 10: Mixed positive and negative float range
    a = np.linspace(-100.0, 100.0, num=50, dtype=np.float32)
    bins = 15
    range_val = [-120.0, 120.0]
    weights = np.ones_like(a, dtype=np.float32)
    list_of_inputs.append({"a": a, "bins": bins, "range": range_val, "weights": weights})

    return list_of_inputs

generated_inputs["jax.numpy.histogram_bin_edges_3"] = histogram_bin_edges_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.histogram_bin_edges_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.histogram_bin_edges_3'.")


check_valid('jax.numpy.histogram_bin_edges', generated_inputs['jax.numpy.histogram_bin_edges_3'], lib="jax", suffix=3)
