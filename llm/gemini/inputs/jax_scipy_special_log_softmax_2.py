
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=(0,)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = (0,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, axis=(1,)
    x = np.random.randn(3, 4).astype(np.float32)
    axis = (1,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, float32, axis=(0,)
    x = np.random.randn(4, 5).astype(np.float32)
    axis = (0,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, float64, axis=(0, 1)
    x = np.random.randn(3, 3).astype(np.float64)
    axis = (0, 1)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float32, axis=(2,)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = (2,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, float32, axis=(0, 2)
    x = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (0, 2)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32, axis=(-1,)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axis = (-1,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative values, float32, axis=(0,)
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
    axis = (0,)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with float64, axis=(-2, -1)
    x = np.random.uniform(-100, 100, (5, 5)).astype(np.float64)
    axis = (-2, -1)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, float32, axis=(1, 3, 4)
    x = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axis = (1, 3, 4)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array with zeros, float32, axis=(0, 1)
    x = np.zeros((3, 3, 3), dtype=np.float32)
    axis = (0, 1)
    input_dict = {"x": x, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
