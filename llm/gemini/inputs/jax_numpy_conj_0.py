
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def conj_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array with negative values
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j, 5.0 - 6.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D complex128 array
    x = np.array(
        [[1.0 - 1.0j, 2.0 + 3.0j], [-4.0 + 0.0j, 0.0 - 5.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 0D scalar array (complex)
    x = np.array(-2.5 + 7.5j, dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D real float32 array
    x = np.array([-1.5, 0.0, 2.5, -3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D real float64 array
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 3D complex64 array
    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Integer array (int32)
    x = np.array([-5, 0, 10, -15], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Integer array (int64, 2D)
    x = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array with complex dtype
    x = np.empty((0, 5), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.conj"] = conj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conj' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conj'.")


check_valid('jax.numpy.conj', generated_inputs['jax.numpy.conj'], lib="jax", suffix=0)
