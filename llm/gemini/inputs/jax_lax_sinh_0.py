
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sinh_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar), float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, float32 with negative and positive values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, float32
    x = np.random.randn(2, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D array, float16
    x = np.random.randn(2, 2, 2, 2).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array, complex64
    x = np.array([1.0 + 1.0j, -2.0 - 0.5j, 3.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array, complex128
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array, float32, special values (inf, -inf, nan, zero)
    x = np.array([-np.inf, -0.0, 0.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D array, float32
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array with 0-size dimension, float32
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array, float32, large magnitude values
    x = np.array([-10.0, -50.0, 50.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.sinh"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sinh'.")


check_valid('jax.lax.sinh', generated_inputs['jax.lax.sinh'], lib="jax", suffix=0)
