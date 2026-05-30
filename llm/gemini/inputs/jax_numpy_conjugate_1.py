
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conjugate_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64 array
    x = np.array([2 - 1j, 3 + 5j, 7], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D complex128 array
    x = np.array([[1 + 2j, -3 - 4j], [5j, -6j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D float32 array with positive and negative numbers
    x = np.array([-1.5, 0.0, 2.3], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D int32 array
    x = np.array([[[1, -2], [3, 4]], [[-5, 6], [7, -8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D complex64 array generated randomly
    x = (np.random.randn(2, 2, 2, 2).astype(np.float32) + 
         1j * np.random.randn(2, 2, 2, 2).astype(np.float32))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float64 with negative numbers and zeros
    x = np.array([-10.0, -5.5, 0.0, 5.5, 10.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex128 purely imaginary array
    x = np.array([[1j, -2j, 3j], [-4j, 5j, -6j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 0D array (scalar tensor) complex64
    x = np.array(1.5 - 2.5j, dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Large 2D float32 array
    x = np.random.randn(100, 100).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.conjugate_1"] = conjugate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.conjugate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.conjugate_1'.")


check_valid('jax.numpy.conjugate', generated_inputs['jax.numpy.conjugate_1'], lib="jax", suffix=1)
