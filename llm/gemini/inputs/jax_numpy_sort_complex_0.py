
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sort_complex_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D complex array (complex64)
    a = np.array([3+4j, 1-2j, 1+2j, 2+0j], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: 1D complex array with negative values (complex128)
    a = np.array([-1-1j, -2+3j, 0+0j, -1+1j], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: 2D real array (will be upcast to complex)
    a = np.array([[5, 3, 4], [6, 9, 2]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: 2D complex array
    a = np.array([[2+3j, 1+1j], [5-1j, 4+2j]], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: 3D complex array to test sorting along the last axis
    a = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    a = a.astype(np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: 1D integer array (will be upcast to complex)
    a = np.array([10, -5, 0, 3, -1], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: 1D array with identical real parts, varying imaginary parts
    a = np.array([2+3j, 2-1j, 2+0j, 2+5j], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: Single element 1D array
    a = np.array([42-42j], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: Large 1D complex array
    a = (np.random.rand(100) + 1j * np.random.rand(100)).astype(np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: 2D array with 1 row
    a = np.array([[3+3j, 1+1j, 2+2j]], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 11: Empty array along sorting axis
    a = np.empty((2, 0), dtype=np.complex64)
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
