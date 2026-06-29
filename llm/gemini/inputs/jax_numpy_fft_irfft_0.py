
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def irfft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 real array, n=4, axis=-1, norm="backward"
    a = np.random.randn(3).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 4,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex64 array, n=5, axis=0, norm="ortho"
    a = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": 0,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D complex64 array, n=6, axis=-1, norm="forward"
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 6,
        "axis": -1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 real array, n=8, axis=0, norm="backward"
    a = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "n": 8,
        "axis": 0,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D complex128 array, n=10, axis=2, norm="ortho"
    a = (np.random.randn(2, 3, 6) + 1j * np.random.randn(2, 3, 6)).astype(np.complex128)
    input_dict = {
        "a": a,
        "n": 10,
        "axis": 2,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 real array, n=5, axis=1, norm="forward"
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "n": 5,
        "axis": 1,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float32 array with negative values, n=6, axis=-1, norm="backward"
    a = np.array([-1.5, 2.3, -0.7, 4.1], dtype=np.float32)
    input_dict = {
        "a": a,
        "n": 6,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D complex64 array, n=12, axis=3, norm="ortho"
    a = (np.random.randn(2, 2, 2, 7) + 1j * np.random.randn(2, 2, 2, 7)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 12,
        "axis": 3,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D complex64 array, n=3, axis=-2, norm="forward"
    a = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex64)
    input_dict = {
        "a": a,
        "n": 3,
        "axis": -2,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D complex128 array, n=7, axis=-1, norm="backward"
    a = (np.random.randn(3, 3, 4) + 1j * np.random.randn(3, 3, 4)).astype(np.complex128)
    input_dict = {
        "a": a,
        "n": 7,
        "axis": -1,
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fft.irfft"] = irfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.irfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.irfft'.")


check_valid('jax.numpy.fft.irfft', generated_inputs['jax.numpy.fft.irfft'], lib="jax", suffix=0)
