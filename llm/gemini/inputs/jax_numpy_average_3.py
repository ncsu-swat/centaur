
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def average_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = [0]
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, returned=True
    a = np.random.randn(4, 3).astype(np.float32)
    axis = [1]
    weights = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, keepdims=True, multi-axis
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = [0, 2]
    weights = np.ones((2, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 with negative values
    a = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
    axis = [0]
    weights = np.array([1.5, 2.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32 array for average, float32 weights
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = [0]
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with reduction over multiple axes
    a = np.random.randn(2, 2, 2).astype(np.float32)
    axis = [1, 2]
    weights = np.ones((2, 2), dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, single axis reduction
    a = np.random.randn(4, 5, 6).astype(np.float32)
    axis = [1]
    weights = np.arange(1, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, full reduction using axis, returned=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = [0, 1]
    weights = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 array, keepdims=True
    a = np.random.randn(6, 2).astype(np.float64)
    axis = [0]
    weights = np.ones((6,), dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array with complete reduction and keepdims=True
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = [0, 1, 2]
    weights = np.random.rand(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.average_3"] = average_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.average_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.average_3'.")


check_valid('jax.numpy.average', generated_inputs['jax.numpy.average_3'], lib="jax", suffix=3)
