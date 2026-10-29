
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 2: 1D array, negative axis
    x = np.array([-1.5, 0.0, 1.5, 2.5], dtype=np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 3: 2D array, axis=1
    x = np.random.randn(3, 5).astype(np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 4: 2D array, axis=0
    x = np.random.randn(4, 6).astype(np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 5: 2D array float64, axis=-1
    x = np.random.randn(5, 5).astype(np.float64)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 6: 3D array, axis=2
    x = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 7: 3D array, negative axis=-2
    x = np.random.randn(2, 4, 3).astype(np.float32)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 8: 4D array, axis=3
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    axis = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 9: 4D array float64, axis=0
    x = np.random.randn(3, 2, 2, 2).astype(np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 10: 2D float16 array, axis=1
    x = np.random.randn(4, 8).astype(np.float16)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 11: Array with extreme negative values
    x = np.array([-1000.0, -500.0, -100.0, 0.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    # Input 12: 5D array, axis=4
    x = np.random.randn(2, 1, 2, 1, 3).astype(np.float32)
    axis = 4
    list_of_inputs.append(copy.deepcopy({"x": x, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.scipy.special.log_softmax_1"] = log_softmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.log_softmax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.log_softmax_1'.")


check_valid('jax.scipy.special.log_softmax', generated_inputs['jax.scipy.special.log_softmax_1'], lib="jax", suffix=1)
