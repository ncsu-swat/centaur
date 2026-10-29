
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def square_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D float64 array
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D) float32 array
    x = np.array(5.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D int32 array with negative integers
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D int64 array
    x = np.random.randint(-100, 100, size=(4, 5)).astype(np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float16 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Complex64 array
    x = np.array([1.0 + 2.0j, -3.0 - 4.0j, 0.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Complex128 2D array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D int32 array with ones and zeros
    x = np.array([0, 1, 1, 0, 1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Empty float32 array
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.square"] = square_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.square' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.square'.")


check_valid('jax.lax.square', generated_inputs['jax.lax.square'], lib="jax", suffix=0)
