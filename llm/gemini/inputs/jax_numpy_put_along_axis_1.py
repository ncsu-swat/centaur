
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, axis=0, mode='clip'
    arr = np.random.randn(3, 4).astype(np.float32)
    indices = np.random.randint(0, 3, size=(1, 4)).astype(np.int32)
    values = np.random.randn(1, 4).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 0,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, mode='drop'
    arr = np.random.randn(3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(3, 1)).astype(np.int32)
    values = np.random.randn(3, 1).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 1,
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, axis=0, mode='clip'
    arr = np.random.randn(5).astype(np.float32)
    indices = np.random.randint(0, 5, size=(2,)).astype(np.int32)
    values = np.random.randn(2).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 0,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=1, mode='fill' (represented as 'fill' or 'clip')
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    indices = np.random.randint(0, 3, size=(2, 1, 4)).astype(np.int64)
    values = np.random.randn(2, 1, 4).astype(np.float64)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 1,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with integers, axis=2, mode='clip'
    arr = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    indices = np.random.randint(0, 4, size=(2, 3, 2)).astype(np.int32)
    values = np.random.randint(100, 200, size=(2, 3, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 2,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, axis=2, mode='drop'
    arr = np.random.randn(2, 2, 3, 3).astype(np.float32)
    indices = np.random.randint(0, 3, size=(2, 2, 1, 3)).astype(np.int32)
    values = np.random.randn(2, 2, 1, 3).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 2,
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, negative axis=-1, mode='clip'
    arr = np.random.randn(5, 5).astype(np.float32)
    indices = np.random.randint(0, 5, size=(5, 2)).astype(np.int32)
    values = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": -1,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, negative axis=-2, mode='drop'
    arr = np.random.randn(5, 5).astype(np.float32)
    indices = np.random.randint(0, 5, size=(2, 5)).astype(np.int32)
    values = np.random.randn(2, 5).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": -2,
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, large size, axis=0, mode='clip'
    arr = np.random.randint(0, 100, size=(100,)).astype(np.int32)
    indices = np.random.randint(0, 100, size=(50,)).astype(np.int32)
    values = np.random.randint(-100, 0, size=(50,)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 0,
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, axis=0, mode='drop'
    arr = np.random.randn(3, 3, 3).astype(np.float32)
    indices = np.random.randint(0, 3, size=(1, 3, 3)).astype(np.int32)
    values = np.random.randn(1, 3, 3).astype(np.float32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": 0,
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_along_axis_1"] = put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_1'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_1'], lib="jax", suffix=1)
