
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=(0,)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 2: 2D array, float32, axis=(1,)
    x = np.random.randn(3, 5).astype(np.float32)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 3: 2D array, float64, axis=(0, 1)
    x = np.random.randn(4, 4).astype(np.float64)
    axis = (0, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 4: 2D array with negative values, float32, axis=(-1,)
    x = -np.abs(np.random.randn(2, 6)).astype(np.float32)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 5: 3D array, float32, axis=(1, 2)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (1, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 6: 3D array, float64, axis=(0,)
    x = np.random.randn(3, 2, 5).astype(np.float64)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 7: 4D array, float32, axis=(-2, -1)
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    axis = (-2, -1)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 8: 1D array, float64, axis=(-1,)
    x = np.linspace(-10.0, 10.0, 20, dtype=np.float64)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 9: 3D array with wide dynamic range, float32, axis=(0, 1, 2)
    x = (np.random.randn(2, 4, 3) * 100.0).astype(np.float32)
    axis = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 10: 4D array, float64, axis=(1, 3)
    x = np.random.randn(3, 4, 2, 5).astype(np.float64)
    axis = (1, 3)
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.scipy.special.log_softmax_2"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.log_softmax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.log_softmax_2'.")


check_valid('jax.scipy.special.log_softmax', generated_inputs['jax.scipy.special.log_softmax_2'], lib="jax", suffix=2)
