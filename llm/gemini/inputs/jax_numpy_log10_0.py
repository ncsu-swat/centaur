
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def log10_inputs():
    list_of_inputs = []

    # Input 1: 1D array of positive float32 numbers
    x = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of positive float64 numbers
    x = np.array([[1.0, 10.0], [100.0, 10000.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array) positive integer
    x = np.array(100, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array with negative numbers (produces nan)
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array of positive floats
    x = np.random.uniform(0.1, 100.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Complex numbers (1D array)
    x = np.array([1.0 + 2.0j, -1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Boolean array
    x = np.array([[True, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Array with infinity and special values
    x = np.array([0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: High-dimensional array (4D) with int64
    x = np.ones((2, 2, 2, 2), dtype=np.int64) * 10
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Complex128 array
    x = np.array([[-10.0 - 5.0j, 10.0 + 5.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 12: Very small positive numbers (subnormal/near zero)
    x = np.array([1e-30, 1e-15, 1e-5], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.log10"] = log10_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log10'.")


check_valid('jax.numpy.log10', generated_inputs['jax.numpy.log10'], lib="jax", suffix=0)
