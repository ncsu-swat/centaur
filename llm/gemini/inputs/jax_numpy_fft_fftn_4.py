
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftn_inputs():
    list_of_inputs = []

    # Input 1: 2D array, float32, basic ortho norm
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": [0, 1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, float32, backward norm
    a = np.random.randn(4, 4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4, 4),
        "axes": [0, 1, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, complex64, forward norm
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": (6, 6),
        "axes": [0, 1],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array (represented as tensor), float64, changing size with s
    a = np.random.randn(10).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (16,),
        "axes": [0],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, float64, subset of axes
    a = np.random.randn(5, 5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "s": (8, 8),
        "axes": [0, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, float32, subset of axes, s smaller than original shape
    a = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (2, 3),
        "axes": [1, 2],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, float32, single axis transform
    a = np.random.randn(12, 12).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (6,),
        "axes": [1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex array, complex128, all axes
    a = (np.random.randn(3, 3, 3) + 1j * np.random.randn(3, 3, 3)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": (3, 3, 3),
        "axes": [0, 1, 2],
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, float32, subset of 3 axes
    a = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4, 4),
        "axes": [0, 2, 4],
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, float32, negative axes indices
    a = np.random.randn(8, 8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": (4, 4),
        "axes": [-2, -1],
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftn_4"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_4'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_4'], lib="jax", suffix=4)
