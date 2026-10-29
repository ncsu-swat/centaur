
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def histogram_bin_edges_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, scalar bins, standard range
    a = np.array([0.5, 1.2, 2.1, 3.7, 4.0], dtype=np.float32)
    bins = np.array(5, dtype=np.int32)
    range_param = (0.0, 5.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, scalar bins, symmetric range
    a = np.random.randn(4, 5).astype(np.float64)
    bins = np.array(10, dtype=np.int64)
    range_param = (-3.0, 3.0)
    weights = np.random.uniform(0.1, 1.0, size=a.shape).astype(np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array, 1D explicit bin edges
    a = np.array([1, 2, 4, 7, 8, 9, 3], dtype=np.int32)
    bins = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32)
    range_param = (0.0, 10.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, scalar bins, wide range
    a = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    bins = np.array(20, dtype=np.int32)
    range_param = (-15.0, 15.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D strictly negative float32 array
    a = np.array([-45.0, -32.5, -20.1, -12.0, -1.5], dtype=np.float32)
    bins = np.array(8, dtype=np.int32)
    range_param = (-50.0, 0.0)
    weights = np.random.rand(5).astype(np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D int64 array, scalar bins
    a = np.random.randint(0, 100, size=(2, 2, 3, 2), dtype=np.int64)
    bins = np.array(15, dtype=np.int32)
    range_param = (0.0, 100.0)
    weights = np.ones_like(a, dtype=np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, 1D explicit monotonically increasing bin edges
    a = np.array([[-8.0, -2.0], [1.0, 7.0]], dtype=np.float32)
    bins = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    range_param = (-10.0, 10.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int32 array, small number of bins
    a = np.array([5, 12, 25, 33, 48], dtype=np.int32)
    bins = np.array(4, dtype=np.int32)
    range_param = (0.0, 50.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float64 array, non-uniform 1D bin edges
    a = np.array([0.15, 0.35, 0.8, 1.5, 1.9], dtype=np.float64)
    bins = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float64)
    range_param = (0.1, 2.0)
    weights = np.ones_like(a, dtype=np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float64 array, large bin count
    a = np.random.randn(3, 3, 3).astype(np.float64)
    bins = np.array(50, dtype=np.int64)
    range_param = (-4.0, 4.0)
    weights = np.random.rand(3, 3, 3).astype(np.float64)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D int32 array, odd number of bins
    a = np.array([-80, -40, 0, 40, 80], dtype=np.int32)
    bins = np.array(7, dtype=np.int32)
    range_param = (-100.0, 100.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D float32 array, range [0, 1]
    a = np.random.rand(5, 5).astype(np.float32)
    bins = np.array(12, dtype=np.int32)
    range_param = (0.0, 1.0)
    weights = np.ones_like(a, dtype=np.float32)
    input_dict = {"a": a, "bins": bins, "range": range_param, "weights": weights}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
