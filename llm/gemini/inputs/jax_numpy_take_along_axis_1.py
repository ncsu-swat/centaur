
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def take_along_axis_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, valid indices, axis=0
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative/out-of-bounds indices, clip mode, axis=0
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([-1, 0, 3], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1, mode="fill"
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([[0, 2], [1, 0]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis=0, with out of bounds and fill with NaN
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    indices = np.array([[1, 0], [2, 1]], dtype=np.int64)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": float('nan')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array with negative axis (-1)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([[2, 1], [0, 2]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": -1,
        "mode": "fill",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=2, fill mode
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2, 3, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "fill_value": 99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis=0, with broadcasting on indices
    arr = np.random.randn(3, 2, 2).astype(np.float32)
    indices = np.random.randint(0, 3, size=(1, 2, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=1, clip mode
    arr = np.random.randn(2, 4, 3).astype(np.float32)
    indices = np.random.randint(-1, 5, size=(2, 2, 3)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "clip",
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float arr with out of bounds, fill_value negative
    arr = np.array([[10., 20.], [30., 40.]], dtype=np.float32)
    indices = np.array([[2, 1], [0, -1]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "fill_value": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, negative axis=-2
    arr = np.random.randn(2, 2, 2, 2).astype(np.float32)
    indices = np.random.randint(0, 2, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": -2,
        "mode": "fill",
        "fill_value": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_along_axis_1"] = take_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_along_axis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_along_axis_1'.")


check_valid('jax.numpy.take_along_axis', generated_inputs['jax.numpy.take_along_axis_1'], lib="jax", suffix=1)
