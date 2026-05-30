
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def searchsorted_inputs():
    list_of_inputs = []

    # Input 1 — Standard float32 1D arrays, left search
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([0.5, 2.0, 2.5, 5.5], dtype=np.float32)
    sorter = np.arange(len(a), dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "left",
        "sorter": sorter,
        "method": "scan"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 — Unsorted int32 with 2D query values, right search
    a = np.array([5, 3, 1, 4, 2, 6], dtype=np.int32)
    v = np.array([[2, 4], [1, 5]], dtype=np.int32)
    sorter = np.argsort(a).astype(np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "right",
        "sorter": sorter,
        "method": "scan_unrolled"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 — float64 3D query values, left search
    a = np.linspace(-10.0, 10.0, 10, dtype=np.float64)
    v = np.random.uniform(-15.0, 15.0, (2, 2, 2)).astype(np.float64)
    sorter = np.arange(len(a), dtype=np.int64)
    input_dict = {
        "a": a,
        "v": v,
        "side": "left",
        "sorter": sorter,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 — Unsorted float32 with negative values, compare_all method
    a = np.array([10, -5, 3, -1, 0, 8, -12, 4], dtype=np.float32)
    v = np.array([-15.0, 0.0, 5.0], dtype=np.float32)
    sorter = np.argsort(a).astype(np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "right",
        "sorter": sorter,
        "method": "compare_all"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 — Large int64 arrays, scan method
    a = np.arange(100, dtype=np.int64)
    v = np.random.randint(-10, 110, (5, 5), dtype=np.int64)
    sorter = np.arange(len(a), dtype=np.int64)
    input_dict = {
        "a": a,
        "v": v,
        "side": "left",
        "sorter": sorter,
        "method": "scan"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 — Very small array, compare_all method
    a = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    v = np.array([-2.0, -1.5, 0.5, 2.0], dtype=np.float32)
    sorter = np.arange(len(a), dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "right",
        "sorter": sorter,
        "method": "compare_all"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 — Unsorted float64, 3D query values, sort method
    a = np.random.randn(15).astype(np.float64)
    v = np.random.randn(2, 3, 2).astype(np.float64)
    sorter = np.argsort(a).astype(np.int64)
    input_dict = {
        "a": a,
        "v": v,
        "side": "left",
        "sorter": sorter,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 — Sorted int32 array with spacing, scan_unrolled method
    a = np.arange(0, 40, 2, dtype=np.int32)
    v = np.array([5, 10, 15, 20, 25], dtype=np.int32)
    sorter = np.arange(len(a), dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "right",
        "sorter": sorter,
        "method": "scan_unrolled"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 — Unsorted float32 with larger search target shape
    a = np.array([9.5, 2.3, -1.1, 4.4, 0.0, -5.5, 8.8, 12.1, -10.0, 3.3, 1.1, -2.2], dtype=np.float32)
    v = np.random.uniform(-12, 15, (3, 4)).astype(np.float32)
    sorter = np.argsort(a).astype(np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "left",
        "sorter": sorter,
        "method": "scan"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 — Large float64 space, sort method
    a = np.linspace(100, 200, 50, dtype=np.float64)
    v = np.random.uniform(90, 210, (2, 2, 2)).astype(np.float64)
    sorter = np.arange(len(a), dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "side": "right",
        "sorter": sorter,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.searchsorted"] = searchsorted_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.searchsorted' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.searchsorted'.")


check_valid('jax.numpy.searchsorted', generated_inputs['jax.numpy.searchsorted'], lib="jax", suffix=0)
