
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def irfft2_inputs():
    list_of_inputs = []

    # Input 1: 2D complex64, default-like parameters
    a = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "s": (4, 8),
        "axes": (-2, -1),
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32, ortho norm
    a = np.random.randn(6, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (6, 6),
        "axes": (-2, -1),
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex128, forward norm
    a = (np.random.randn(2, 5, 5) + 1j * np.random.randn(2, 5, 5)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "s": (5, 8),
        "axes": (1, 2),
        "norm": "forward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64, non-consecutive axes
    a = np.random.randn(8, 3, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (0, 2),
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D complex64, middle axes
    a = (
        np.random.randn(2, 4, 7, 3) + 1j * np.random.randn(2, 4, 7, 3)
    ).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (4, 12),
        "axes": (1, 2),
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64, odd size output along transformed axis
    a = (np.random.randn(7, 4) + 1j * np.random.randn(7, 4)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "s": (7, 7),
        "axes": (0, 1),
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32, axes with negative indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (3, 8),
        "axes": (-3, -1),
        "norm": "forward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex64, first two axes
    a = (np.random.randn(4, 5, 2) + 1j * np.random.randn(4, 5, 2)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "s": (6, 9),
        "axes": (0, 1),
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D float64, last two axes
    a = np.random.randn(1, 2, 2, 8, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": (-2, -1),
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex128, larger dimension
    a = (np.random.randn(16, 17) + 1j * np.random.randn(16, 17)).astype(
        np.complex128
    )
    input_dict = {
        "a": a,
        "s": (16, 32),
        "axes": (0, 1),
        "norm": "forward",
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
