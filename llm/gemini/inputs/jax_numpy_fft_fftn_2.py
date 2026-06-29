
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftn_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, full transform, backward norm
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex64 array, truncated shape, ortho norm
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (2, 3),
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, full transform, forward norm
    a = np.random.randn(3, 3, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (3, 3, 3),
        "axes": (0, 1, 2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array, zero-padded, backward norm
    a = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (16,),
        "axes": (0,),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex128 array, transform along last two axes, ortho norm
    a = (np.random.randn(2, 4, 4) + 1j * np.random.randn(2, 4, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": (1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, transform along non-contiguous axes, forward norm
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (2, 5),
        "axes": (0, 3),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array with negative values, backward norm
    a = np.random.uniform(-10.0, -1.0, (5, 5)).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (5, 5),
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, custom shape and axes, ortho norm
    a = np.random.randn(5, 5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (3, 2, 4),
        "axes": (0, 1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex64 array, forward norm
    a = (np.random.randn(16) + 1j * np.random.randn(16)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (8,),
        "axes": (0,),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 array, axes swapped, backward norm
    a = np.random.randn(6, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": (1, 0),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftn_2"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_2'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_2'], lib="jax", suffix=2)
