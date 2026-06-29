
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfft2_inputs():
    list_of_inputs = []

    # Case 1: Simple 2D complex64, even output shape
    a = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [8, 8],
        "axes": [-2, -1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D complex128, custom axes, ortho norm
    a = (np.random.randn(5, 6, 4) + 1j * np.random.randn(5, 6, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": [6, 6],
        "axes": [1, 2],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 2D float32 input (will be cast to complex internally), forward norm
    a = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [3, 4],
        "axes": [0, 1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D complex64, backward norm, axes=(1, 2)
    a = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": [1, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D float64 input, ortho norm
    a = np.random.randn(6, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [6, 8],
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 3D complex64, odd output shape on the last axis
    a = (np.random.randn(4, 4, 3) + 1j * np.random.randn(4, 4, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [4, 5],
        "axes": [1, 2],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: 3D complex128, non-contiguous axes (0, 2)
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": [3, 8],
        "axes": [0, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 8: 2D float32 input with large output size
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [8, 14],
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 9: 4D complex64, transforms on the last two axes
    a = (np.random.randn(2, 2, 4, 3) + 1j * np.random.randn(2, 2, 4, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": [2, 3],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 10: 2D complex64, larger dimensions
    a = (np.random.randn(10, 10) + 1j * np.random.randn(10, 10)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [10, 18],
        "axes": [-2, -1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.irfft2_2"] = irfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfft2_2'.")


check_valid('jax.numpy.fft.irfft2', generated_inputs['jax.numpy.fft.irfft2_2'], lib="jax", suffix=2)
