
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tanh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, 2D random array
    x = np.random.randn(3, 4).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 3D random array
    x = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64, 1D array
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float32, large negative values
    x = np.array([-100.0, -50.0, -10.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 4D random array
    x = np.random.randn(2, 2, 2, 2).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex128, 2D random array
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32, single element array
    x = np.array([0.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float32, multidimensional array with singleton dimensions
    x = np.random.randn(1, 5, 1, 5).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32, values close to zero
    x = np.array([-1e-5, 0.0, 1e-5], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.tanh"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tanh'.")


check_valid('jax.lax.tanh', generated_inputs['jax.lax.tanh'], lib="jax", suffix=0)
