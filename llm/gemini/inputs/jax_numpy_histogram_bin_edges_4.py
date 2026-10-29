
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, basic range and bins
    a = np.array([1.0, 2.5, 3.2, 4.8, 5.0], dtype=np.float32)
    bins = 5
    range_arr = np.array([0.0, 6.0], dtype=np.float32)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, larger bin count
    a = np.random.randint(0, 100, size=(50,), dtype=np.int32)
    bins = 10
    range_arr = np.array([0.0, 100.0], dtype=np.float64)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, negative to positive range
    a = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float64)
    bins = 8
    range_arr = np.array([-1.0, 1.0], dtype=np.float64)
    weights = np.random.uniform(0.1, 1.0, size=(4, 5)).astype(np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D strictly negative values
    a = np.array([-10.0, -8.5, -4.2, -1.1], dtype=np.float32)
    bins = 4
    range_arr = np.array([-12.0, 0.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    a = np.random.randn(2, 3, 4).astype(np.float32)
    bins = 16
    range_arr = np.array([-3.0, 3.0], dtype=np.float32)
    weights = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single bin case
    a = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    bins = 1
    range_arr = np.array([0.0, 10.0], dtype=np.float32)
    weights = np.ones(3, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array with wide range
    a = np.array([[-30, -10, 0], [10, 25, 50]], dtype=np.int64)
    bins = 12
    range_arr = np.array([-50.0, 50.0], dtype=np.float64)
    weights = np.ones_like(a, dtype=np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float64 with large number of bins
    a = np.linspace(-500.0, 500.0, num=100, dtype=np.float64)
    bins = 50
    range_arr = np.array([-1000.0, 1000.0], dtype=np.float64)
    weights = np.random.rand(100).astype(np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 tensor
    a = np.random.uniform(0.0, 10.0, size=(2, 2, 2, 3)).astype(np.float32)
    bins = 7
    range_arr = np.array([0.0, 10.0], dtype=np.float32)
    weights = np.ones((2, 2, 2, 3), dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element array
    a = np.array([3.14], dtype=np.float32)
    bins = 3
    range_arr = np.array([1.0, 5.0], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D uint8 array with full byte range
    a = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    bins = 20
    range_arr = np.array([0.0, 255.0], dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_arr, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
