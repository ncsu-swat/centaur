
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array with default backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": [-2, -1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64 array with ortho norm
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array with forward norm
    a = np.random.randn(2, 6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (6, 6),
        "axes": [1, 2],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Size expansion (padding)
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (6, 6),
        "axes": [0, 1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array transforming specific axes
    a = np.random.randn(2, 3, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": [2, 3],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Size reduction (cropping)
    a = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (5, 5),
        "axes": [0, 1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D complex128 array
    a = (np.random.randn(3, 8, 8) + 1j * np.random.randn(3, 8, 8)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": [-2, -1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Non-square dimensions
    a = np.random.randn(16, 16).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 12),
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Custom axes non-contiguous
    a = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (2, 2),
        "axes": [0, 2],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher-dimensional complex array
    a = (np.random.randn(2, 2, 8, 8) + 1j * np.random.randn(2, 2, 8, 8)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": [-2, -1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifft2_4"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_4'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_4'], lib="jax", suffix=4)
