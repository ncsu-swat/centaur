
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def searchsorted_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, left side, scan method
    a = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    sorter = np.argsort(a)
    v = np.array([0.0, 2.0, 5.0, 8.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "scan",
        }
    )

    # Input 2: Right side, scan_unrolled method, integer array
    a = np.array([10, 20, 20, 30, 40], dtype=np.int32)
    sorter = np.arange(len(a), dtype=np.int32)
    v = np.array([20, 25], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "right",
            "sorter": sorter,
            "method": "scan_unrolled",
        }
    )

    # Input 3: 2D v array, sort method
    a = np.array([-5.0, -2.0, 0.0, 3.0, 9.0], dtype=np.float64)
    sorter = np.argsort(a)
    v = np.array([[-3.0, 0.0], [4.0, 10.0]], dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "sort",
        }
    )

    # Input 4: compare_all method, small array
    a = np.array([1, 4, 9, 16], dtype=np.int64)
    sorter = np.array([0, 1, 2, 3], dtype=np.int64)
    v = np.array([0, 1, 5, 9, 20], dtype=np.int64)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "right",
            "sorter": sorter,
            "method": "compare_all",
        }
    )

    # Input 5: Unsorted `a` handled by non-trivial `sorter`
    a = np.array([50, 10, 40, 20, 30], dtype=np.int32)
    sorter = np.argsort(a).astype(np.int32)
    v = np.array([15, 35, 55], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "scan",
        }
    )

    # Input 6: Scalar/0D query value
    a = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    sorter = np.argsort(a)
    v = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "right",
            "sorter": sorter,
            "method": "scan",
        }
    )

    # Input 7: 3D query values
    a = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    sorter = np.arange(len(a), dtype=np.int32)
    v = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "scan_unrolled",
        }
    )

    # Input 8: Repeated elements in a with right side
    a = np.array([2, 2, 2, 2, 2], dtype=np.int32)
    sorter = np.arange(len(a), dtype=np.int32)
    v = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "right",
            "sorter": sorter,
            "method": "sort",
        }
    )

    # Input 9: Repeated elements in a with left side
    a = np.array([2, 2, 2, 2, 2], dtype=np.int32)
    sorter = np.arange(len(a), dtype=np.int32)
    v = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "compare_all",
        }
    )

    # Input 10: Float64 precision with unsorted a and large 1D query
    a = np.array([3.14, 2.71, -1.0, 0.0, 1.41], dtype=np.float64)
    sorter = np.argsort(a).astype(np.int64)
    v = np.linspace(-2.0, 4.0, 20, dtype=np.float64)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "side": "left",
            "sorter": sorter,
            "method": "scan",
        }
    )

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
