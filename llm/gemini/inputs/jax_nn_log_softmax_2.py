
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    x = np.array([1.0, 2.0, -1.0, 0.0, 3.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, False, True, True], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 2: 2D float32 array, axis 1
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    where = np.random.choice([True, False], size=(3, 4), p=[0.8, 0.2])
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 3: 2D float64 array, axis 0
    x = np.random.randn(4, 3).astype(np.float64)
    axis = (0,)
    where = np.ones((4, 3), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 4: 3D array, axis 2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (2,)
    where = np.random.choice([True, False], size=(2, 3, 4), p=[0.9, 0.1])
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 5: 3D array with multiple axes in tuple
    x = np.random.randn(3, 3, 3).astype(np.float32)
    axis = (1, 2)
    where = np.ones((3, 3, 3), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 6: 4D array, negative values, axis -1
    x = np.random.uniform(-10.0, 0.0, (2, 2, 3, 3)).astype(np.float32)
    axis = (-1,)
    where = np.random.choice([True, False], size=(2, 2, 3, 3), p=[0.7, 0.3])
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 7: 2D array, broad-castable where mask
    x = np.random.randn(5, 5).astype(np.float32)
    axis = (1,)
    where = np.array([[True], [False], [True], [True], [False]], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 8: Large magnitude values
    x = np.array([[100.0, 99.0, 98.0], [-100.0, -101.0, -102.0]], dtype=np.float32)
    axis = (1,)
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 9: 5D array, axis tuple (2, 3, 4)
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (2, 3, 4)
    where = np.random.choice([True, False], size=(2, 2, 2, 2, 2), p=[0.85, 0.15])
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    # Input 10: 1D array with positive values only
    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    axis = (0,)
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append({"x": x, "axis": axis, "where": where})

    return list_of_inputs

generated_inputs["jax.nn.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.log_softmax_2'.")


check_valid('jax.nn.log_softmax', generated_inputs['jax.nn.log_softmax_2'], lib="jax", suffix=2)
