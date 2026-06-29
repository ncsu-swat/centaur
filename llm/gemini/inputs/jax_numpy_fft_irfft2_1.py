
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfft2_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D complex input
    a = (np.random.randn(8, 5) + 1j * np.random.randn(8, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (-2, -1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D complex input, ortho norm
    a = (np.random.randn(2, 4, 4) + 1j * np.random.randn(2, 4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (4, 6),
        "axes": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex input with custom axes
    a = (np.random.randn(10, 6, 3) + 1j * np.random.randn(10, 6, 3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (10, 10),
        "axes": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Odd sizes for s
    a = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (3, 3),
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D input, complex128
    a = (np.random.randn(2, 5, 3, 4) + 1j * np.random.randn(2, 5, 3, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": (5, 5),
        "axes": (1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Real-valued input (valid for irfft2)
    a = np.random.randn(12, 7).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (12, 12),
        "axes": (-2, -1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D input with non-adjacent axes
    a = (np.random.randn(6, 4, 5) + 1j * np.random.randn(6, 4, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (6, 8),
        "axes": (0, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D complex128, ortho
    a = (np.random.randn(16, 9) + 1j * np.random.randn(16, 9)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": (16, 16),
        "axes": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Real-valued float64 3D input
    a = np.random.randn(3, 8, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (8, 4),
        "axes": (1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D complex input
    a = (np.random.randn(2, 2, 4, 6) + 1j * np.random.randn(2, 2, 4, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (4, 10),
        "axes": (2, 3),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.irfft2_1"] = irfft2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfft2_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfft2_1'.")


check_valid('jax.numpy.fft.irfft2', generated_inputs['jax.numpy.fft.irfft2_1'], lib="jax", suffix=1)
