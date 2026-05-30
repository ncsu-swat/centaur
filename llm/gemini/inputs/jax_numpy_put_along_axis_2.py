
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 2D array, updating along axis 1
    arr = np.array([[10, 30, 20], [60, 40, 50]], dtype=np.float32)
    indices = np.array([[1], [0]], dtype=np.int32)
    values = 99.0
    axis = 1
    inplace = False
    mode = "clip"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 2: 2D array, updating along axis 0
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    indices = np.array([[0, 2, 1]], dtype=np.int32)
    values = -1.0
    axis = 0
    inplace = False
    mode = "drop"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 3: 1D array
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    values = 5.5
    axis = 0
    inplace = False
    mode = "clip"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 4: 3D array, updating along axis 1
    arr = np.ones((2, 3, 4), dtype=np.float32)
    indices = np.ones((2, 1, 4), dtype=np.int32)
    values = 0.0
    axis = 1
    inplace = False
    mode = "drop"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 5: 2D float64 array, negative axis index
    arr = np.random.randn(4, 4).astype(np.float64)
    indices = np.array([[0], [1], [2], [3]], dtype=np.int32)
    values = 12.34
    axis = -1
    inplace = False
    mode = "promise_in_bounds"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 6: 1D array with larger index set
    arr = np.arange(10, dtype=np.float32)
    indices = np.array([0, 2, 4, 6, 8], dtype=np.int32)
    values = -999.0
    axis = 0
    inplace = False
    mode = "clip"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 7: 3D array, updating along axis 2
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    indices = np.zeros((2, 2, 1), dtype=np.int32)
    values = 1.0
    axis = 2
    inplace = False
    mode = "drop"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 8: 2D array, updating along axis 0
    arr = np.random.randn(5, 2).astype(np.float32)
    indices = np.array([[2, 3]], dtype=np.int32)
    values = 42.0
    axis = 0
    inplace = False
    mode = "clip"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 9: 1D float64 array with int64 indices
    arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    indices = np.array([2], dtype=np.int64)
    values = -3.14
    axis = -1
    inplace = False
    mode = "drop"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    # Input 10: 4D array, updating along axis 1
    arr = np.ones((2, 2, 2, 2), dtype=np.float32)
    indices = np.zeros((2, 1, 2, 2), dtype=np.int32)
    values = 100.0
    axis = 1
    inplace = False
    mode = "clip"
    list_of_inputs.append({
        "arr": arr,
        "indices": indices,
        "values": values,
        "axis": axis,
        "inplace": inplace,
        "mode": mode
    })

    return list_of_inputs

generated_inputs["jax.numpy.put_along_axis_2"] = put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_2'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_2'], lib="jax", suffix=2)
