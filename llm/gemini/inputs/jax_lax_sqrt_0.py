
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, accuracy None
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, accuracy None
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64 array, accuracy None
    x = np.random.uniform(1.0, 100.0, size=(2, 2, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D complex64 array, accuracy None
    x = np.array([-4.0 + 0j, 0.0 + 0j, 4.0 + 3j], dtype=np.complex64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D float16 array, accuracy None
    x = np.random.uniform(0.1, 5.0, size=(2, 2, 2, 2)).astype(np.float16)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex128 array, accuracy None
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(np.complex128)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large 1D float32 array, accuracy None
    x = np.linspace(0.01, 100.0, 100).astype(np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar-like 0D float32 array, accuracy None
    x = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array with negative values, accuracy None
    x = np.array([[[-1.0, 2.0], [3.0, -4.0]]], dtype=np.float32)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float64 array, accuracy None
    x = np.random.uniform(10.0, 20.0, size=(1, 2, 1, 3, 2)).astype(np.float64)
    input_dict = {"x": x, "accuracy": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.sqrt"] = sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sqrt' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sqrt'.")


check_valid('jax.lax.sqrt', generated_inputs['jax.lax.sqrt'], lib="jax", suffix=0)
