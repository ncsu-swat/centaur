
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isnan_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32 containing NaN, Inf, and normal numbers
    x = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of float64 with no NaNs
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array of float16 containing NaNs
    x = np.array([[[np.nan, 1.0], [2.0, np.nan]], [[3.0, 4.0], [np.nan, np.nan]]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D complex array (complex64) with NaN real/imaginary parts
    x = np.array([1 + 2j, np.nan + 3j, 4 + np.nan*1j, np.nan + np.nan*1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar) float32 with NaN
    x = np.array(np.nan, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of float32 with random numbers
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex128 array
    x = np.array([[1.0 + 1.0j, np.nan + 0.0j], [np.inf - 1j, -np.nan * 1j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array of int32 (valid, should return all False)
    x = np.array([-10, 0, 5, 100], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large 2D array of float32 with sparse NaNs
    x = np.random.randn(50, 50).astype(np.float32)
    x[12, 15] = np.nan
    x[34, 42] = np.nan
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D boolean array (valid input type)
    x = np.array([True, False, True, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 5D array of float64
    x = np.zeros((2, 2, 2, 2, 2), dtype=np.float64)
    x[1, 0, 1, 0, 1] = np.nan
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isnan_1"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isnan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isnan_1'.")


check_valid('jax.numpy.isnan', generated_inputs['jax.numpy.isnan_1'], lib="jax", suffix=1)
