
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, axis=1, clip mode
    arr = np.random.randn(3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(3, 1)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(99),
        "axis": int(1),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=0, clip mode
    arr = np.random.randint(-10, 10, size=(5, 2)).astype(np.int32)
    indices = np.random.randint(0, 5, size=(1, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(-1),
        "axis": int(0),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, axis=0, drop mode
    arr = np.random.randn(10).astype(np.float64)
    indices = np.random.randint(0, 10, size=(3,)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(42),
        "axis": int(0),
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, axis=2, drop mode
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2, 3, 1)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(0),
        "axis": int(2),
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=0, clip mode
    arr = np.random.randint(0, 100, size=(2, 3, 4)).astype(np.int64)
    indices = np.random.randint(0, 2, size=(1, 3, 4)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(5),
        "axis": int(0),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, same shape for indices, axis=1, drop mode
    arr = np.random.randn(4, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(4, 4)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(-100),
        "axis": int(1),
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, axis=3, clip mode
    arr = np.random.randn(2, 2, 2, 2).astype(np.float32)
    indices = np.random.randint(0, 2, size=(2, 2, 2, 1)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(7),
        "axis": int(3),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Column vector array, axis=0, clip mode
    arr = np.random.randn(6, 1).astype(np.float32)
    indices = np.random.randint(0, 6, size=(1, 1)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(123),
        "axis": int(0),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, negative axis=-1, drop mode
    arr = np.random.randint(-5, 5, size=(3, 5)).astype(np.int32)
    indices = np.random.randint(0, 5, size=(3, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(8),
        "axis": int(-1),
        "inplace": False,
        "mode": "drop"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, negative axis=-2, clip mode
    arr = np.random.randn(2, 4, 3).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2, 1, 3)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "values": int(-9),
        "axis": int(-2),
        "inplace": False,
        "mode": "clip"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.put_along_axis_3"] = jax_numpy_put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_3'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_3'], lib="jax", suffix=3)
