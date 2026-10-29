
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_conj_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar complex64
    x = np.array(1.0 - 2.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar complex128
    x = np.array(-3.5 + 4.2j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array complex64
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j, 5.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array complex128 with negative values
    x = np.array([-1.5 - 2.5j, -0.0 + 0.0j, 3.14 - 1.59j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array complex64
    x = np.array(
        [[1.0 + 1.0j, 2.0 - 2.0j], [-3.0 + 3.0j, -4.0 - 4.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array complex128 with special values (inf, nan)
    x = np.array(
        [
            [np.nan + 1.0j, 1.0 + np.inf * 1j],
            [-np.inf - 1.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array complex64
    x = (
        np.random.randn(2, 3, 4).astype(np.float32)
        + 1j * np.random.randn(2, 3, 4).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array complex128
    x = (
        np.random.randn(2, 2, 3, 2) + 1j * np.random.randn(2, 2, 3, 2)
    ).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 5D array complex64
    x = (
        np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
        + 1j * np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array complex64
    x = np.array([], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Purely imaginary array complex128
    x = np.array([[0.0 + 5.0j, 0.0 - 3.0j], [0.0 + 1.0j, 0.0 - 0.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.conj"] = lax_conj_inputs()

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
