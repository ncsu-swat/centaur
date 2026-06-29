
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ifft2_inputs():
    list_of_inputs = []

    # Input 1: Standard 2D real float32 array
    input_dict = {
        "a": np.random.randn(4, 4).astype(np.float32),
        "s": (4, 4),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64 array with crop/pad (s < shape)
    input_dict = {
        "a": (np.random.randn(8, 8) + 1j * np.random.randn(8, 8)).astype(np.complex64),
        "s": (6, 6),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D real float32 array with ortho norm
    input_dict = {
        "a": np.random.randn(3, 5, 5).astype(np.float32),
        "s": (5, 5),
        "axes": (1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex64 array with axes (-2, -1)
    input_dict = {
        "a": (np.random.randn(2, 4, 6) + 1j * np.random.randn(2, 4, 6)).astype(np.complex64),
        "s": (4, 4),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array with pad (s > shape)
    input_dict = {
        "a": np.random.randn(10, 10).astype(np.float64),
        "s": (12, 12),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, transforming middle dimensions
    input_dict = {
        "a": np.random.randn(2, 4, 4, 2).astype(np.float32),
        "s": (3, 3),
        "axes": (1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array containing negative values
    input_dict = {
        "a": -np.abs(np.random.randn(5, 5)).astype(np.float32),
        "s": (5, 5),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array transforming the first two axes
    input_dict = {
        "a": np.random.randn(4, 3, 2).astype(np.float32),
        "s": (4, 3),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 2D complex128 array
    input_dict = {
        "a": (np.random.randn(16, 16) + 1j * np.random.randn(16, 16)).astype(np.complex128),
        "s": (16, 16),
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32 array with cropping
    input_dict = {
        "a": np.random.randn(2, 10, 10).astype(np.float32),
        "s": (8, 8),
        "axes": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.ifft2_2"] = ifft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.ifft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.ifft2_2'.")


check_valid('jax.numpy.fft.ifft2', generated_inputs['jax.numpy.fft.ifft2_2'], lib="jax", suffix=2)
