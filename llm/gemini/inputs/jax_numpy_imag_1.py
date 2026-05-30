
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_imag_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array (standard case)
    val = np.array([1 + 2j, 3 - 4j, 5 + 0j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D complex128 array
    val = np.array([[1j, 2 - 3j], [-4j, 5 + 6j]], dtype=np.complex128)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array (real numbers only, should yield 0s)
    val = np.array([-1.5, 0.0, 3.14], dtype=np.float32)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex64 array
    val = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D complex64 array (scalar wrapped in array)
    val = np.array(3 - 7j, dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array (integers, should yield 0s)
    val = np.array([[1, -2], [3, 4]], dtype=np.int32)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex128 array with random values
    val = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex128)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex64 array with purely imaginary numbers
    val = np.array([5j, -2.5j, 0j, 11j], dtype=np.complex64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float64 array
    val = np.array([[1.1, 2.2, -3.3], [4.4, -5.5, 6.6]], dtype=np.float64)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D complex128 array with large/small values
    val = np.array([1e10 + 2e10j, 1e-10 - 2e-10j], dtype=np.complex128)
    input_dict = {"val": val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.imag_1"] = jax_numpy_imag_inputs()

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
