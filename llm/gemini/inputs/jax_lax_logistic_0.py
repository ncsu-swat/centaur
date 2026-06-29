
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logistic_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive/negative values
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float64, standard normal
    x = np.random.randn(3, 3).astype(np.float64)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D array (scalar-like), float32
    x = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "accuracy": "FASTEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, complex64
    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex64)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, float32, large positive/negative values
    x = np.random.uniform(-100.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, float32, zeros
    x = np.zeros((10,), dtype=np.float32)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, complex128
    x = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex128)
    input_dict = {"x": x, "accuracy": "FASTEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D array, float16
    x = np.random.randn(2, 2, 1, 3, 2).astype(np.float16)
    input_dict = {"x": x, "accuracy": "DEFAULT"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, float32, extremely small and large values
    x = np.array([-1e5, -1e-5, 0.0, 1e-5, 1e5], dtype=np.float32)
    input_dict = {"x": x, "accuracy": "HIGHEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, float64, all negatives
    x = -np.abs(np.random.randn(2, 3, 4)).astype(np.float64)
    input_dict = {"x": x, "accuracy": "FASTEST"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.logistic"] = logistic_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.logistic' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.logistic'.")


check_valid('jax.lax.logistic', generated_inputs['jax.lax.logistic'], lib="jax", suffix=0)
