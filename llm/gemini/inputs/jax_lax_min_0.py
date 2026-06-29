
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def min_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    y = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays
    x = np.random.randn(3, 4).astype(np.float32)
    y = np.random.randn(3, 4).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 arrays
    x = np.random.randn(2, 3, 4).astype(np.float64)
    y = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D int32 arrays
    x = np.array([-10, 20, 30], dtype=np.int32)
    y = np.array([5, 15, -25], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 arrays
    x = np.random.randint(-100, 100, size=(5, 5)).astype(np.int64)
    y = np.random.randint(-100, 100, size=(5, 5)).astype(np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D arrays (scalars) float32
    x = np.array(-5.5, dtype=np.float32)
    y = np.array(2.3, dtype=np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex64 arrays
    x = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    y = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting with same rank: 2D float32 (1, 3) and (4, 3)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.random.randn(4, 3).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcasting with same rank: 3D int32 (2, 1, 3) and (2, 4, 3)
    x = np.random.randint(-10, 10, size=(2, 1, 3)).astype(np.int32)
    y = np.random.randint(-10, 10, size=(2, 4, 3)).astype(np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large 2D arrays float32
    x = np.random.randn(128, 128).astype(np.float32)
    y = np.random.randn(128, 128).astype(np.float32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex128 arrays
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    y = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex128)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.min"] = min_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.min' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.min'.")


check_valid('jax.lax.min', generated_inputs['jax.lax.min'], lib="jax", suffix=0)
