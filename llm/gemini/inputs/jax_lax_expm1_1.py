
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, small values near zero
    x = np.array([-1e-5, 0.0, 1e-5, 0.1, -0.1], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64, negative and positive values
    x = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32, random values
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float16, very small values
    x = np.array([-1e-3, 1e-3, 0.0], dtype=np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D array (scalar) float32
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64, all negative values
    x = np.random.uniform(-10.0, -1.0, size=(2, 2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D complex128, near zero
    x = np.array([1e-5 + 1e-5j, -1e-5 + 1e-5j], dtype=np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64, mixed values including large ones
    x = np.array([[[10.0, -20.0], [0.1, -0.1]], [[0.0, 0.0], [5.0, -5.0]]], dtype=np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32, large negative values (tending to -1)
    x = np.array([[-100.0, -50.0], [-20.0, -10.0]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.expm1_1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expm1_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expm1_1'.")


check_valid('jax.lax.expm1', generated_inputs['jax.lax.expm1_1'], lib="jax", suffix=1)
