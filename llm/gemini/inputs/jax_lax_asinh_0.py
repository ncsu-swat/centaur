
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: float32 1D array, positive values
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 1D array, negative and zero values
    x = np.array([-10.0, -5.0, -1.0, 0.0], dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64 2D array, mixed values
    x = np.array([[-2.5, -0.5], [0.5, 2.5]], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 3D array, random values
    x = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 1D array
    x = np.array([1.0 + 1.0j, -2.0 - 3.0j, 0.0 + 0.0j], dtype=np.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 2D array
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 0D array (scalar array)
    x = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 4D array, large size
    x = np.random.uniform(-100.0, 100.0, size=(2, 2, 3, 3)).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 1D array with very small values
    x = np.array([1e-15, -1e-15, 1e-30], dtype=np.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 5D array
    x = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.asinh"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.asinh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.asinh'.")


check_valid('jax.lax.asinh', generated_inputs['jax.lax.asinh'], lib="jax", suffix=0)
