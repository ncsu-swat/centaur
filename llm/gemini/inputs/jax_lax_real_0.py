
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_real_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar complex64
    x = np.array(3.0 - 4.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array complex64
    x = np.array(
        [-1.5 + 2.5j, 0.0 + 0.0j, 4.2 - 1.1j, -3.3 - 5.5j], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array complex128
    x = np.array(
        [[1.0 + 2.0j, -3.0 - 4.0j], [5.0 + 0.0j, 0.0 - 6.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array complex64
    x = (
        np.random.randn(2, 3, 4).astype(np.float32)
        + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array complex128
    x = (
        np.random.randn(2, 2, 3, 2).astype(np.float64)
        + 1j * np.random.randn(2, 2, 3, 2).astype(np.float64)
    ).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array with special values (inf, nan) complex64
    x = np.array(
        [
            complex(np.inf, 1.0),
            complex(-np.inf, -2.0),
            complex(np.nan, 3.0),
            complex(4.0, np.nan),
        ],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Empty 1D array complex64
    x = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty 2D array complex128
    x = np.empty((3, 0), dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Single-element 1D array complex128
    x = np.array([7.5 - 8.2j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D array with pure imaginary and pure real numbers complex64
    x = np.array(
        [[0.0 + 5.0j, 2.0 + 0.0j], [-7.0 + 0.0j, 0.0 - 9.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array complex64
    x = (
        np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
        + 1j * np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.real"] = lax_real_inputs()

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
