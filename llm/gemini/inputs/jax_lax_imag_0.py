
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def imag_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar complex64
    x = np.array(3.0 + 4.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array complex64 with negative imaginary parts
    x = np.array([1.0 - 2.0j, -3.0 + 4.0j, 0.0 - 0.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array complex128
    x = np.array([[1.5 + 2.5j, -3.5 - 4.5j], [0.0 + 1.0j, 2.0 + 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array complex64
    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array complex128
    x = (np.random.randn(1, 2, 2, 3) + 1j * np.random.randn(1, 2, 2, 3)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array with special values (inf, nan) complex64
    x = np.array([np.nan + 1.0j, 1.0 + np.nan * 1j, np.inf - np.inf * 1j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Empty 1D array complex128
    x = np.empty((0,), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 2D array complex64
    x = np.empty((0, 5), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array with purely imaginary numbers complex64
    x = np.array([[0.0 + 5.0j, 0.0 - 10.0j], [0.0 + 0.0j, 0.0 + 1.23j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array with purely real numbers complex128 (imaginary part 0)
    x = np.array([[5.0 + 0.0j, -10.0 + 0.0j], [0.0 + 0.0j, 1.23 + 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array complex64
    x = (np.random.randn(2, 1, 2, 1, 3) + 1j * np.random.randn(2, 1, 2, 1, 3)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

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
