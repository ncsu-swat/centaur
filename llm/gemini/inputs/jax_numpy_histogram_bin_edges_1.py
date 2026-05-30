
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D array, 5 bins
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    bins = 5
    range_val = (1.0, 5.0)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Random normal float32, 10 bins, weights
    a = np.random.randn(10).astype(np.float32)
    bins = 10
    range_val = (-3.0, 3.0)
    weights = np.random.rand(10).astype(np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 2D array (will be flattened by the API)
    a = np.random.randn(5, 5).astype(np.float64)
    bins = 3
    range_val = (-2.0, 2.0)
    weights = np.random.rand(5, 5).astype(np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer inputs
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    bins = 2
    range_val = (0, 50)
    weights = np.array([1, 2, 1, 1], dtype=np.int32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large range with float32 uniform distribution
    a = np.random.uniform(-100, 100, size=(100,)).astype(np.float32)
    bins = 20
    range_val = (-100.0, 100.0)
    weights = np.ones((100,), dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 array
    a = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    bins = 4
    range_val = (1.0, 5.0)
    weights = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float16)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array inputs
    a = np.random.randn(2, 3, 4).astype(np.float32)
    bins = 8
    range_val = (-4.0, 4.0)
    weights = np.random.rand(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Constant value array
    a = np.zeros(10, dtype=np.float32)
    bins = 5
    range_val = (-1.0, 1.0)
    weights = np.ones(10, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer array
    a = np.arange(100).astype(np.int64)
    bins = 50
    range_val = (0, 100)
    weights = np.ones(100).astype(np.int64)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative and positive float32 values
    a = np.random.randn(50).astype(np.float32)
    bins = 15
    range_val = (-5.0, 5.0)
    weights = np.random.randn(50).astype(np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_val, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.histogram_bin_edges_1"] = generate_histogram_bin_edges_inputs()

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
