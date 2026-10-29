
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def irfft_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64, default-like parameters
    a = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    input_dict = {"a": a, "n": 8, "axis": -1, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex128, odd output length
    a = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex128)
    input_dict = {"a": a, "n": 7, "axis": 0, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array along last axis
    a = np.random.randn(3, 6).astype(np.float32)
    input_dict = {"a": a, "n": 10, "axis": -1, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex64 array along first axis
    a = (np.random.randn(5, 4) + 1j * np.random.randn(5, 4)).astype(np.complex64)
    input_dict = {"a": a, "n": 8, "axis": 0, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex128 array along middle axis
    a = (np.random.randn(2, 7, 4) + 1j * np.random.randn(2, 7, 4)).astype(
        np.complex128
    )
    input_dict = {"a": a, "n": 12, "axis": 1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array along negative axis
    a = np.random.randn(4, 5, 6).astype(np.float64)
    input_dict = {"a": a, "n": 8, "axis": -2, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex64 array along axis 2
    a = (
        np.random.randn(2, 3, 9, 2) + 1j * np.random.randn(2, 3, 9, 2)
    ).astype(np.complex64)
    input_dict = {"a": a, "n": 16, "axis": 2, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array, small odd output length
    a = np.random.randn(3).astype(np.float32)
    input_dict = {"a": a, "n": 5, "axis": -1, "norm": "ortho"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64, output length smaller than input
    a = (np.random.randn(4, 8) + 1j * np.random.randn(4, 8)).astype(np.complex64)
    input_dict = {"a": a, "n": 4, "axis": 1, "norm": "forward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D complex128 array along negative leading axis
    a = (
        np.random.randn(5, 2, 3, 4) + 1j * np.random.randn(5, 2, 3, 4)
    ).astype(np.complex128)
    input_dict = {"a": a, "n": 8, "axis": -4, "norm": "backward"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.irfft"] = irfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfft'.")


check_valid('jax.numpy.fft.irfft', generated_inputs['jax.numpy.fft.irfft'], lib="jax", suffix=0)
