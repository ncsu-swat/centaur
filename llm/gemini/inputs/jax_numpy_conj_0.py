
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conj_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array with positive and negative components
    x = np.array([1 + 2j, -3 + 4j, 5 - 6j, -7 - 8j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D complex128 array
    x = np.array([[1.5 - 2.5j, -3.0 + 0j], [0j, 4.5 + 5.5j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D complex64 array
    x = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D complex64 scalar array
    x = np.array(2.0 - 3.0j, dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D float32 array
    x = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D float64 array with negative values
    x = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D int32 array
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Larger 2D complex64 array
    x = (np.random.randn(10, 10) + 1j * np.random.randn(10, 10)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D complex128 array containing nan and inf
    x = np.array([np.nan + 1j, 2.0 - np.inf * 1j, np.inf + np.nan * 1j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D complex64 array
    x = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

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
