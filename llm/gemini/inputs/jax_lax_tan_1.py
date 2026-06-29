
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive and negative values
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array
    x = np.random.uniform(-1.0, 1.0, size=(3, 3)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array
    x = np.random.normal(0.0, 2.0, size=(2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D complex64 array
    x = np.array([1.0 + 1.0j, -1.0 - 2.0j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D complex128 array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array
    x = np.random.uniform(-0.5, 0.5, size=(1, 2, 2, 1)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D (scalar-like) float32 array
    x = np.array(0.785398, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 array with negative values
    x = np.array([[-10.0, -5.0], [-2.0, -1.0]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D complex64 array
    x = (np.random.uniform(-1.0, 1.0, size=(2, 1, 3)) + 1j * np.random.uniform(-1.0, 1.0, size=(2, 1, 3))).astype(np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float64 array with larger range
    x = np.linspace(-3.0, 3.0, 10).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.tan_1"] = jax_lax_tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_1'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_1'], lib="jax", suffix=1)
