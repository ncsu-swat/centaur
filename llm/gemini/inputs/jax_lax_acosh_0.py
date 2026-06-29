
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def acosh_inputs():
    list_of_inputs = []

    # Input 1: float32 1D array, values >= 1.0
    x = np.array([1.0, 1.5, 2.0, 10.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: float64 2D array, values >= 1.0
    x = np.random.uniform(1.0, 50.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": x})

    # Input 3: float16 3D array, values >= 1.0
    x = np.random.uniform(1.0, 10.0, size=(2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": x})

    # Input 4: complex64 1D array
    x = (np.random.uniform(-10, 10, size=5) + 1j * np.random.uniform(-10, 10, size=5)).astype(np.complex64)
    list_of_inputs.append({"x": x})

    # Input 5: complex128 2D array
    x = (np.random.uniform(-5, 5, size=(3, 4)) + 1j * np.random.uniform(-5, 5, size=(3, 4))).astype(np.complex128)
    list_of_inputs.append({"x": x})

    # Input 6: float32 0D array (scalar)
    x = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 7: float64 1D array with values close to 1.0
    x = np.array([1.00001, 1.0001, 1.001], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 8: float32 4D array, values >= 1.0
    x = np.random.uniform(1.0, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": x})

    # Input 9: complex64 3D array
    x = (np.random.uniform(-2, 2, size=(2, 2, 2)) + 1j * np.random.uniform(-2, 2, size=(2, 2, 2))).astype(np.complex64)
    list_of_inputs.append({"x": x})

    # Input 10: float32 2D array, large values
    x = np.random.uniform(100.0, 10000.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.lax.acosh"] = acosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.acosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.acosh'.")


check_valid('jax.lax.acosh', generated_inputs['jax.lax.acosh'], lib="jax", suffix=0)
