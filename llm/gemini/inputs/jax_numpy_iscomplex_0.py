
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def iscomplex_inputs():
    list_of_inputs = []

    # Input 1: 1D complex array with mixed real/complex values
    x = np.array([1.0, 1.0 + 1.0j, 2.0j, 0.0 + 0.0j, -3.0j]).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D complex64 array
    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int32 array
    x = np.array([-10, 0, 10, 20]).astype(np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar) complex
    x = np.array(1 + 2j).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D complex128 array with negative and mixed values
    x = np.array([[-1-1j, -2j], [3, 4+0j]]).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float64 array with np.nan and np.inf
    x = np.array([np.nan, np.inf, -np.inf, 0.0]).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D complex64 array
    x = (np.random.randn(2, 3, 1, 2) + 1j * np.random.randn(2, 3, 1, 2)).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D boolean array
    x = np.array([True, False, True]).astype(np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Empty complex array
    x = np.array([], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.iscomplex"] = iscomplex_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplex' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplex'.")


check_valid('jax.numpy.iscomplex', generated_inputs['jax.numpy.iscomplex'], lib="jax", suffix=0)
