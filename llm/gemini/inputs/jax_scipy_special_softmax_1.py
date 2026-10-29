
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, axis=0
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 2: 1D array with negative values, float64, axis=-1
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 3: 2D array, float32, axis=0
    x = np.random.randn(4, 5).astype(np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 4: 2D array, float32, axis=1
    x = np.random.randn(3, 8).astype(np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 5: 2D array with large magnitude values, float64, axis=-1
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]], dtype=np.float64)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 6: 3D array, float32, axis=0
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 7: 3D array, float32, axis=1
    x = np.random.randn(3, 6, 2).astype(np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 8: 3D array, float64, axis=2
    x = np.random.randn(4, 4, 4).astype(np.float64)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 9: 4D array, float32, axis=-2
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 10: 5D array, float32, axis=3
    x = np.random.randn(2, 2, 3, 4, 2).astype(np.float32)
    axis = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 11: 2D array, float16, axis=0
    x = np.random.randn(5, 5).astype(np.float16)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    return list_of_inputs

generated_inputs["jax.scipy.special.softmax_1"] = softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.softmax_1'.")


check_valid('jax.scipy.special.softmax', generated_inputs['jax.scipy.special.softmax_1'], lib="jax", suffix=1)
