
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def digitize_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, increasing bins, right=False, scan
    x = np.array([0.2, 1.5, 2.7, 3.1, -0.5], dtype=np.float32)
    bins = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "scan",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, increasing bins, right=True, compare_all
    x = np.array([[1, 5, 10], [2, 7, 12]], dtype=np.int32)
    bins = np.array([2, 5, 8, 11], dtype=np.int32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "compare_all",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array with negative values, increasing bins, right=False, sort
    x = np.array([-10.0, -5.5, -0.1, 0.0, 4.2], dtype=np.float64)
    bins = np.array([-8.0, -4.0, 0.0, 4.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, increasing bins, right=True, scan_unrolled
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    bins = np.linspace(-1.0, 1.0, num=5, dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "scan_unrolled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array, decreasing bins, right=False, compare_all
    x = np.array([3.5, 2.1, 1.0, 0.5, -1.0], dtype=np.float32)
    bins = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "compare_all",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, decreasing bins, right=True, scan
    x = np.array([[10.5, 5.0], [0.0, -3.2]], dtype=np.float64)
    bins = np.array([10.0, 5.0, 0.0, -5.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "scan",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D scalar float32 array, increasing bins, right=False, compare_all
    x = np.array(2.5, dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "compare_all",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int64 array, increasing bins, right=False, scan
    x = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    bins = np.array([10, 20, 30, 40], dtype=np.int64)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "scan",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, increasing bins, right=True, scan_unrolled
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    bins = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "scan_unrolled",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with boundary values, increasing bins, right=False, sort
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int16 array, decreasing bins, right=True, scan
    x = np.array([[50, 20], [5, -10]], dtype=np.int16)
    bins = np.array([60, 40, 20, 0], dtype=np.int16)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "scan",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.digitize"] = digitize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.digitize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.digitize'.")


check_valid('jax.numpy.digitize', generated_inputs['jax.numpy.digitize'], lib="jax", suffix=0)
