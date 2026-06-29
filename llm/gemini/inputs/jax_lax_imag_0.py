
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def imag_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D complex128
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D complex64
    x = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D complex128
    x = (np.random.randn(2, 5) + 1j * np.random.randn(2, 5)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D complex64, containing zeros and negatives
    real = np.array([[[0.0, -1.5], [2.3, -4.1]]], dtype=np.float32)
    imag = np.array([[[1.2, 0.0], [-3.4, 5.6]]], dtype=np.float32)
    x = (real + 1j * imag).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D complex128
    x = (np.random.randn(2, 2, 3, 3) + 1j * np.random.randn(2, 2, 3, 3)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 0D (scalar) complex64
    x = np.array(1.5 - 2.5j, dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D complex128 with zero real parts
    x = (1j * np.random.randn(10)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D complex64 with zero imaginary parts
    x = np.random.randn(10).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D complex128
    x = (np.random.randn(1, 2, 1, 3, 2) + 1j * np.random.randn(1, 2, 1, 3, 2)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.imag"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.imag'.")


check_valid('jax.lax.imag', generated_inputs['jax.lax.imag'], lib="jax", suffix=0)
