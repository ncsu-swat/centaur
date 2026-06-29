
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array, standard axes
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [8, 8],
        "axes": [0, 1],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float32 array, ortho norm, transforming last two axes
    a = np.random.randn(4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": [1, 2],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, padding with s, forward norm
    a = np.random.randn(10, 10).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [12, 12],
        "axes": [0, 1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex64 array, transforming non-contiguous axes
    a = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [3, 5],
        "axes": [0, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array, ortho norm on middle axes
    a = np.random.randn(5, 5, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": [1, 2],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, negative axis indexing
    a = np.random.randn(16, 16).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [8, 8],
        "axes": [-2, -1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, cropped output shape
    a = np.random.randn(4, 8, 12).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [6, 10],
        "axes": [1, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 3D float32 array, first two axes
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 2],
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex128 array, forward norm
    a = (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": [10, 10],
        "axes": [0, 1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, wide axes spacing
    a = np.random.randn(2, 4, 6, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [4, 6],
        "axes": [1, 3],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fft2_4"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_4'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_4'], lib="jax", suffix=4)
