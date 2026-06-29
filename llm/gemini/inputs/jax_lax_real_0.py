
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def real_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D complex128 array
    x = np.array([[1.5 - 1.5j, 2.5 + 3.5j], [-0.5 + 0.5j, 0.0 + 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 3D complex64 array
    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 0D complex128 array (scalar-like)
    x = np.array(3.14 - 2.71j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D complex64 array
    x = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex128 array with very small and large values
    x = np.array([1e-15 + 1e15j, -1e15 - 1e-15j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex64 array with purely imaginary numbers
    x = np.array([[0.0 + 1.0j, 0.0 - 2.0j], [0.0 + 3.0j, 0.0 + 0.0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D complex128 array containing NaNs and Infs
    x = np.array([complex(np.nan, 1.0), complex(2.0, np.inf), complex(np.inf, np.nan)], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D complex64 array
    x = (np.random.randn(1, 2, 1, 3, 2) + 1j * np.random.randn(1, 2, 1, 3, 2)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D complex128 array, larger size
    x = (np.random.randn(50, 50) + 1j * np.random.randn(50, 50)).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["jax.lax.real"] = real_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.real' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.real'.")


check_valid('jax.lax.real', generated_inputs['jax.lax.real'], lib="jax", suffix=0)
