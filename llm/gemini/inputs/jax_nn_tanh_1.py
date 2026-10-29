
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64 with negative and positive values
    x = np.array([[-1.0, 0.0, 1.0], [3.0, -2.0, 5.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (float32)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float32
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array of float16
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array of integers (promotes to inexact dtype)
    x = np.array([-10, -1, 0, 1, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex64 array
    x = np.array([[1.0 + 2.0j, -2.0 + 0.5j], [0.0 - 1.0j, 3.0 - 4.0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D complex128 array
    x = np.array([2.0 - 5.0j, -1.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large magnitude values in float32
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.tanh_1"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.tanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.tanh_1'.")


check_valid('jax.nn.tanh', generated_inputs['jax.nn.tanh_1'], lib="jax", suffix=1)
