
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive, negative, and zero float32 values
    x = np.array([2.0, -4.0, 0.0, -1.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with very small values close to zero (float32)
    x = np.array([1e-4, 1e-6, 2e-10, -1e-5, -3e-8], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of float64 values
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D scalar array
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of float16 values
    x = np.random.uniform(-2.0, 2.0, size=(2, 3, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D integer array
    x = np.array([-2, -1, 0, 1, 2, 5], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D array of float32 values
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Complex64 numbers
    x = np.array([0.1 + 0.2j, -0.5 + 1.0j, 0.0 - 0.3j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Complex128 numbers in a 2D array
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(
        np.complex128
    )
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Boolean array
    x = np.array([True, False, True, False], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 12: 1D array with special float values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.expm1_1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_1'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_1'], lib="jax", suffix=1)
