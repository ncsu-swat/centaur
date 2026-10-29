
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with common angles
    x = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with negative values
    x = np.array([-np.pi, -np.pi / 2, -np.pi / 4, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float32 array
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D float64 array
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(2, 3, 5)).astype(
        np.float64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 0D scalar array
    x = np.array(np.pi / 3, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: Integer array (promoted to inexact type by JAX)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: Float16 array
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex64 array
    x = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex128 2D array
    x = np.array(
        [[1.0j, 2.0 - 1.0j], [-1.0 + 0.0j, 0.5 + 0.5j]], dtype=np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 4D array
    x = np.random.randn(2, 2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sin_1"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_1'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_1'], lib="jax", suffix=1)
