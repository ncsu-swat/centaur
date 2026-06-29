
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fftn_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, s=[4, 4], axes=(0, 1), norm="backward"
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [4, 4],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with different size, s=[2, 3], axes=(0, 1), norm="ortho"
    a = np.random.randn(3, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 3],
        "axes": (0, 1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array, s=[2, 2, 2], axes=(0, 1, 2), norm="forward"
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 2, 2],
        "axes": (0, 1, 2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array, s=[5], axes=(0,), norm="backward"
    a = np.random.randn(8).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [5],
        "axes": (0,),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array, transforming over 2 axes, s=[3, 4], axes=(1, 2), norm="ortho"
    a = np.random.randn(2, 5, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "s": [3, 4],
        "axes": (1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, transforming over non-contiguous axes, s=[2, 2], axes=(0, 3), norm="forward"
    a = np.random.randn(3, 4, 4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 2],
        "axes": (0, 3),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64 array, s=[5, 5], axes=(0, 1), norm="backward"
    a = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(np.complex64)
    input_dict = {
        "a": a,
        "s": [5, 5],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex128 array, s=[2, 3, 4], axes=(0, 1, 2), norm="ortho"
    a = (np.random.randn(4, 4, 4) + 1j * np.random.randn(4, 4, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "s": [2, 3, 4],
        "axes": (0, 1, 2),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array with negative values, s=[3, 2], axes=(0, 1), norm="backward"
    a = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    input_dict = {
        "a": a,
        "s": [3, 2],
        "axes": (0, 1),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array, s=[2, 2, 2], axes=(1, 2, 3), norm="forward"
    a = np.random.randn(2, 3, 3, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "s": [2, 2, 2],
        "axes": (1, 2, 3),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.fftn_3"] = fftn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.fftn_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.fftn_3'.")


check_valid('jax.numpy.fft.fftn', generated_inputs['jax.numpy.fft.fftn_3'], lib="jax", suffix=3)
