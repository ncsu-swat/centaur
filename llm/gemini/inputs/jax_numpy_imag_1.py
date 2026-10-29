
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def imag_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    val = np.array([1.0 + 2.0j, -3.0 + 4.5j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 2: 2D complex128 array with negative/zero/inf/nan
    val = np.array(
        [[1.0 - 5.0j, np.nan + 2.0j], [-np.inf + 0.0j, 3.5 - np.inf * 1j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 3: 0D scalar array (complex64)
    val = np.array(4.0 - 7.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 4: 1D real float32 array
    val = np.array([-2.5, 0.0, 3.14, -100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 5: 3D complex64 array
    val = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 6: 2D integer array (int32)
    val = np.array([[1, -2, 3], [0, 5, -6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 7: Empty 1D complex array
    val = np.array([], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 8: 4D float64 array
    val = np.random.randn(1, 2, 2, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 9: 2D int64 array
    val = np.array([[10, -20], [-30, 40]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 10: 0D float64 scalar array
    val = np.array(-42.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"val": val}))

    # Input 11: 3D complex128 array
    val = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"val": val}))

    return list_of_inputs


generated_inputs["jax.numpy.imag_1"] = imag_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.imag_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.imag_1'.")


check_valid('jax.numpy.imag', generated_inputs['jax.numpy.imag_1'], lib="jax", suffix=1)
