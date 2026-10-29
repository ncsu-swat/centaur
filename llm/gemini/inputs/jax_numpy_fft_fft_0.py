
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, backward normalization
    a = np.array([1.0, 2.0, -3.0, 4.0, -5.0, 6.0], dtype=np.float32)
    input_dict = {"a": a, "n": 6, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64 array, ortho normalization along axis 0
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
    input_dict = {"a": a, "n": 8, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array with padding (n > axis length), forward normalization
    a = np.random.randn(3, 5).astype(np.float64)
    input_dict = {"a": a, "n": 10, "axis": 1, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D complex128 array with truncation (n < axis length)
    a = (np.arange(16) + 2j * np.arange(16)).astype(np.complex128)
    input_dict = {"a": a, "n": 8, "axis": 0, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis index
    a = np.random.uniform(-10.0, 10.0, size=(2, 4, 6)).astype(np.float32)
    input_dict = {"a": a, "n": 6, "axis": -2, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex64 array, transform along last axis
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    input_dict = {"a": a, "n": 4, "axis": 2, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D integer array
    a = np.array([-10, 0, 5, 20, -15, 3], dtype=np.int32)
    input_dict = {"a": a, "n": 6, "axis": -1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "n": 7, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element along transformed axis (n=1)
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {"a": a, "n": 1, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float64 array, forward normalization
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float64)
    input_dict = {"a": a, "n": 5, "axis": 3, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.fft"] = fft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft'.")


check_valid('jax.numpy.fft.fft', generated_inputs['jax.numpy.fft.fft'], lib="jax", suffix=0)
