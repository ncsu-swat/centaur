
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensorinv_inputs():
    list_of_inputs = []

    # Input 1: 2D array, ind=1, float32
    a = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"a": a, "ind": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array, ind=2, float32
    a = np.random.randn(2, 2, 4).astype(np.float32)
    input_dict = {"a": a, "ind": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, ind=2, float64
    a = np.random.randn(3, 2, 6).astype(np.float64)
    input_dict = {"a": a, "ind": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, ind=2, float32
    a = np.random.randn(2, 3, 2, 3).astype(np.float32)
    input_dict = {"a": a, "ind": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, ind=1, float64
    a = np.random.randn(8, 2, 4).astype(np.float64)
    input_dict = {"a": a, "ind": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 6D array, ind=3, float32
    a = np.random.randn(2, 2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "ind": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, ind=1, complex64
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    input_dict = {"a": a, "ind": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Minimal dimension, ind=1, float32
    a = np.random.randn(1, 1).astype(np.float32)
    input_dict = {"a": a, "ind": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, ind=1, float32
    a = np.random.randn(16, 4, 4).astype(np.float32)
    input_dict = {"a": a, "ind": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, ind=2, float64
    a = np.random.randn(3, 4, 2, 6).astype(np.float64)
    input_dict = {"a": a, "ind": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.tensorinv"] = tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensorinv'.")


check_valid('jax.numpy.linalg.tensorinv', generated_inputs['jax.numpy.linalg.tensorinv'], lib="jax", suffix=0)
