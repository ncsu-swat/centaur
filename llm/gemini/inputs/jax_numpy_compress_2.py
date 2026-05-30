
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def compress_inputs():
    list_of_inputs = []

    # Input 1: Basic boolean mask along axis 0
    condition = np.array([True, False, True], dtype=bool)
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    axis = 0
    size = 2
    fill_value = 0.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 2: Integer condition along axis 1
    condition = np.array([1, 0, 1, 1], dtype=np.int32)
    a = np.arange(12).reshape(3, 4).astype(np.float32)
    axis = 1
    size = 3
    fill_value = -1.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 3: Condition length matches axis size, size is equal to axis size
    condition = np.array([True, True], dtype=bool)
    a = np.arange(6).reshape(2, 3).astype(np.float32)
    axis = 0
    size = 2
    fill_value = 9.9
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 4: 3D array with axis=2 (and condition truncation)
    condition = np.array([True, False, True, False, True], dtype=bool)
    a = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    axis = 2
    size = 2
    fill_value = 0.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 5: Condition is shorter than axis dimension
    condition = np.array([True, False], dtype=bool)
    a = np.arange(12).reshape(4, 3).astype(np.float32)
    axis = 0
    size = 2
    fill_value = -999.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 6: 1D array compression
    condition = np.array([True, True, True, True], dtype=bool)
    a = np.arange(4).astype(np.float32)
    axis = 0
    size = 4
    fill_value = 1.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 7: 3D random array along axis 1
    condition = np.array([True, False, True], dtype=bool)
    a = np.random.randn(3, 3, 3).astype(np.float32)
    axis = 1
    size = 2
    fill_value = 0.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 8: 1D array with single element output
    condition = np.array([0, 0, 1], dtype=np.int32)
    a = np.random.randn(3).astype(np.float32)
    axis = 0
    size = 1
    fill_value = 0.5
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 9: Floating-point condition (nonzero elements are True)
    condition = np.array([1.0, 0.0, 2.0, 3.0, 0.0], dtype=np.float32)
    a = np.arange(20).reshape(5, 4).astype(np.float32)
    axis = 0
    size = 3
    fill_value = -1.0
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    # Input 10: Negative axis index
    condition = np.array([True], dtype=bool)
    a = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    axis = -1
    size = 1
    fill_value = 3.14
    list_of_inputs.append({
        "condition": condition,
        "a": a,
        "axis": axis,
        "size": size,
        "fill_value": fill_value
    })

    return list_of_inputs

generated_inputs["jax.numpy.compress_2"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_2'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_2'], lib="jax", suffix=2)
