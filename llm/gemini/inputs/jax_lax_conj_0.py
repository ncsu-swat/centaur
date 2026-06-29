
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conj_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D complex64 array
    x = np.array([[1.0 - 1.0j, 2.0 + 2.0j], [-3.0 + 3.0j, -4.0 - 4.0j]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D complex128 array
    x = np.random.randn(2, 3, 4).astype(np.float64) + 1j * np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D complex128 array
    x = np.array([0.0 + 0.0j, -1.0j, 1.0], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D complex64 array
    x = np.random.randn(2, 2, 2, 2).astype(np.float32) + 1j * np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D complex64 array (scalar)
    x = np.array(5.0 - 3.0j, dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex128 array
    x = np.random.randn(10, 10).astype(np.float64) + 1j * np.random.randn(10, 10).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 3D complex64 array with inf/nan values
    real_part = np.array([[[1.0, np.inf], [np.nan, -2.0]]], dtype=np.float32)
    imag_part = np.array([[[np.nan, -np.inf], [3.0, 4.0]]], dtype=np.float32)
    x = real_part + 1j * imag_part
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D complex64 array with all zeros
    x = np.zeros((5,), dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D complex64 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32) + 1j * np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.conj"] = conj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conj'.")


check_valid('jax.lax.conj', generated_inputs['jax.lax.conj'], lib="jax", suffix=0)
