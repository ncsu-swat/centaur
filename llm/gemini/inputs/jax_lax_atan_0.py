
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atan_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array with negative, zero, and positive values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with random values
    x = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar (0D array) float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float32 array with large values
    x = np.random.uniform(-100.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float32 array with infinities and nan
    x = np.array([-np.inf, np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D complex128 array
    real_part = np.random.randn(2, 2)
    imag_part = np.random.randn(2, 2)
    x = (real_part + 1j * imag_part).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D float32 array
    x = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float64 array with very small values
    x = np.array([1e-15, -1e-15, 1e-30], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 3D complex64 array
    real_part = np.random.randn(2, 2, 2)
    imag_part = np.random.randn(2, 2, 2)
    x = (real_part + 1j * imag_part).astype(np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.atan"] = atan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.atan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.atan'.")


check_valid('jax.lax.atan', generated_inputs['jax.lax.atan'], lib="jax", suffix=0)
