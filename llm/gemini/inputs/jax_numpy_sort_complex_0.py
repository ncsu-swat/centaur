
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_complex_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array with mixed signs
    a = np.array([1 + 2j, -2 + 4j, 3 - 1j, -2 - 3j], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 1D complex128 array with identical real parts to test tie-breaking
    a = np.array([2 + 5j, 2 - 1j, 2 + 0j, 2 + 3j], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 2D complex64 array
    a = np.array(
        [[5 + 1j, 3 - 2j, 4 + 0j], [-6 + 3j, 9 - 1j, 2 + 2j]],
        dtype=np.complex64,
    )
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 2D float32 array (will be upcast to complex)
    a = np.array([[5.0, 3.0, 4.0], [6.0, 9.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 3D complex64 array
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 1D int32 array (integers converted to complex)
    a = np.array([-10, 5, 0, -2, 100, 3], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 1D float64 array with negative values
    a = np.array([-1.5, 2.3, -0.4, 0.0, 10.2, -8.7], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: 4D complex128 array
    a = (np.random.randn(2, 2, 3, 4) + 1j * np.random.randn(2, 2, 3, 4)).astype(
        np.complex128
    )
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: 1D boolean array
    a = np.array([True, False, True, False, True], dtype=np.bool_)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 1D empty complex array
    a = np.array([], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 11: 2D int64 array with negative numbers
    a = np.array([[-3, 4, 1], [0, -5, -2]], dtype=np.int64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 12: 1D complex128 array with pure imaginary numbers
    a = np.array([0 + 3j, 0 - 2j, 0 + 0j, 0 + 1j], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs


generated_inputs["jax.numpy.sort_complex"] = sort_complex_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_complex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_complex'.")


check_valid('jax.numpy.sort_complex', generated_inputs['jax.numpy.sort_complex'], lib="jax", suffix=0)
