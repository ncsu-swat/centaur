
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_sin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex64 array
    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D (scalar) float32 array
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float32 array
    x = np.random.uniform(-np.pi, np.pi, (2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D complex128 array with negative values
    x = np.array([-2.0 - 3.0j, -0.5 + 0.5j, 1.0 - 1.0j], dtype=np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array with small values
    x = np.array([[1e-5, -1e-5], [2e-5, -2e-5]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float64 array with large values
    x = np.array([[[100.0, -100.0]], [[50.0, -50.0]]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array with zeros and pi multiples
    x = np.array([0.0, np.pi, -np.pi, 2 * np.pi], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D complex64 array with mixed signs
    x = np.array([[1.5 + 2.5j, -1.5 - 2.5j], [0.0 + 0.0j, -0.5 + 0.0j]], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.sin_2"] = jax_lax_sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sin_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sin_2'.")


check_valid('jax.lax.sin', generated_inputs['jax.lax.sin_2'], lib="jax", suffix=2)
