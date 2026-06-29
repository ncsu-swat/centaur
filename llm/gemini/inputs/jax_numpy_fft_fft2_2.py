
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fft2_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, standard backward norm
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [8, 8],
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64, ortho norm
    a = (np.random.randn(16, 16) + 1j * np.random.randn(16, 16)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [16, 16],
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, forward norm, specifying negative axes
    a = np.random.randn(3, 6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex128, custom sizes (padding/truncating)
    a = (np.random.randn(2, 10, 10) + 1j * np.random.randn(2, 10, 10)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": [6, 12],
        "axes": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32, ortho norm on inner dimensions
    a = np.random.randn(2, 2, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [3, 5],
        "axes": (2, 3),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32, forward norm (implicit casting to float/complex)
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    input_dict = {
        "a": a,
        "s": [5, 5],
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32, spanning non-adjacent axes
    a = np.random.randn(4, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [10, 10],
        "axes": (0, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D complex64, custom shapes and axes
    a = (np.random.randn(2, 5, 2, 5) + 1j * np.random.randn(2, 5, 2, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [8, 4],
        "axes": (1, 3),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 with negative values
    a = np.random.uniform(-50.0, -10.0, (10, 10)).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [7, 9],
        "axes": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D float32, smaller transform size
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 2],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fft2_2"] = fft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fft2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fft2_2'.")


check_valid('jax.numpy.fft.fft2', generated_inputs['jax.numpy.fft.fft2_2'], lib="jax", suffix=2)
