
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def average_inputs():
    list_of_inputs = []

    # Input 1: 2D array, single axis, returned=False, keepdims=False
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    weights = np.random.rand(4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, single axis, returned=True, keepdims=True
    a = np.random.randn(3, 4).astype(np.float32)
    axis = (0,)
    weights = np.random.rand(3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, multiple axes, returned=False, keepdims=False
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (1, 2)
    weights = np.random.rand(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, multiple axes, returned=True, keepdims=True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (0, 2)
    weights = np.random.rand(2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array, returned=False, keepdims=True
    a = np.random.randn(5).astype(np.float32)
    axis = (0,)
    weights = np.random.rand(5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, matching weight shape, returned=True, keepdims=False
    a = np.random.randn(3, 3).astype(np.float32)
    axis = (0, 1)
    weights = np.random.rand(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, subset axes, returned=False, keepdims=True
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = (2, 3)
    weights = np.random.rand(2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative values, float64, returned=True, keepdims=False
    a = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = (1,)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, full reduction, returned=False, keepdims=False
    a = np.random.randn(3, 2, 5).astype(np.float32)
    axis = (0, 1, 2)
    weights = np.random.rand(3, 2, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, float64, returned=True, keepdims=True
    a = np.random.randn(6).astype(np.float64)
    axis = (0,)
    weights = np.random.rand(6).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": axis,
        "weights": weights,
        "returned": True,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.average_2"] = average_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.average_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.average_2'.")


check_valid('jax.numpy.average', generated_inputs['jax.numpy.average_2'], lib="jax", suffix=2)
