
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def digitize_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D array, increasing bins, right=False
    x = np.array([1.0, 2.0, 2.5, 1.5, 3.0, 3.5], dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float32 1D array, increasing bins, right=True
    x = np.array([1.0, 2.0, 2.5, 1.5, 3.0, 3.5], dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, increasing bins, right=False
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, decreasing bins, right=True
    x = np.random.uniform(0, 10, (2, 3, 4)).astype(np.float32)
    bins = np.array([8.0, 6.0, 4.0, 2.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Integer array, increasing integer bins, right=False
    x = np.array([1, 5, 10, 15, 20], dtype=np.int32)
    bins = np.array([5, 10, 15], dtype=np.int32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large float64 1D array, increasing bins, right=True
    x = np.linspace(-10.0, 10.0, 100).astype(np.float64)
    bins = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative values, increasing bins, right=False
    x = np.array([-3.5, -2.1, -0.5, 0.2, 1.8], dtype=np.float32)
    bins = np.array([-3.0, -1.0, 1.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 1-element array, increasing bins, right=False
    x = np.array([2.3], dtype=np.float32)
    bins = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional 4D float32 array, increasing bins, right=True
    x = np.random.uniform(-5.0, 5.0, (2, 2, 2, 2)).astype(np.float32)
    bins = np.array([-2.5, 0.0, 2.5], dtype=np.float32)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": True,
        "method": "sort"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Decreasing integer bins, int64 array, right=False
    x = np.array([100, 50, 0, -50], dtype=np.int64)
    bins = np.array([80, 40, 0, -40], dtype=np.int64)
    input_dict = {
        "x": x,
        "bins": bins,
        "right": False,
        "method": "sort"
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
