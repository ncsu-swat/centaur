
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis=(0,), simple mask
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True, False, True], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 2: 2D float32 array, axis=(1,), random mask
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.random.choice([True, False], size=(3, 4), p=[0.8, 0.2]).astype(bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 3: 2D float64 array, axis=(0,), all True mask
    x = np.random.randn(4, 2).astype(np.float64)
    axis = (0,)
    where = np.ones((4, 2), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 4: 3D float32 array, multi-axis tuple=(1, 2)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (1, 2)
    where = np.random.choice([True, False], size=(2, 3, 4)).astype(bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 5: Negative values, axis=(-1,)
    x = -np.abs(np.random.randn(5, 5).astype(np.float32))
    axis = (-1,)
    where = np.ones((5, 5), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 6: Large values, axis=(0, 1)
    x = (np.random.randn(3, 3) * 100).astype(np.float32)
    axis = (0, 1)
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 7: Small values, axis=(1,)
    x = (np.random.randn(2, 5) - 100).astype(np.float32)
    axis = (1,)
    where = np.ones((2, 5), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 8: High-dimensional 4D array, axis=(2, 3)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (2, 3)
    where = np.random.choice([True, False], size=(2, 2, 3, 3)).astype(bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 9: Broadcastable mask shape, axis=(1,)
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.array([[True], [False], [True]], dtype=bool)  # shape (3, 1) broadcastable to (3, 4)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 10: Float16 array, axis=(0,)
    x = np.random.randn(4, 4).astype(np.float16)
    axis = (0,)
    where = np.random.choice([True, False], size=(4, 4)).astype(bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    return list_of_inputs

generated_inputs["jax.nn.softmax_2"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softmax_2'.")


check_valid('jax.nn.softmax', generated_inputs['jax.nn.softmax_2'], lib="jax", suffix=2)
