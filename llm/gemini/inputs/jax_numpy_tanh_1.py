
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64 with negative and positive values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of float16
    x = np.random.randn(2, 3, 5).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D integer array (jax.numpy.tanh promotes to inexact dtype)
    x = np.array([-10, -2, 0, 2, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array of float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex64 array
    x = np.array([1.0 + 2.0j, -2.0 + 1.0j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 2D complex128 array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Array with large magnitudes (approaching saturation +/- 1)
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Array with special float values (inf, -inf, nan)
    x = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.tanh_1"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tanh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tanh_1'.")


check_valid('jax.numpy.tanh', generated_inputs['jax.numpy.tanh_1'], lib="jax", suffix=1)
