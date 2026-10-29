
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cosh_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, and zero float32
    x = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array float32
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 4D array float16
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D complex64 array
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D complex128 array
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Empty array float32
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large magnitude values float64
    x = np.array([-50.0, -20.0, 20.0, 50.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D array float32
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array with special values float32
    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.cosh"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cosh'.")


check_valid('jax.lax.cosh', generated_inputs['jax.lax.cosh'], lib="jax", suffix=0)
