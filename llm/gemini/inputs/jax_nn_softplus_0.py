
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def softplus_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array (float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with positive, negative, and zero values (float32)
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array standard normal values (float32)
    x = np.random.randn(4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array float64 precision
    x = np.random.randn(3, 3).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array (float32)
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D batch array representing feature maps (float32)
    x = np.random.randn(2, 8, 8, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D array with large positive numbers to test numerical stability
    x = np.array([50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 1D array with large negative numbers
    x = np.array([-1000.0, -500.0, -100.0, -50.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 2D array float16 precision
    x = np.array([[-2.0, -0.5], [0.5, 2.0]], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty 1D array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 5D array
    x = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs

generated_inputs["jax.nn.softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.softplus'.")


check_valid('jax.nn.softplus', generated_inputs['jax.nn.softplus'], lib="jax", suffix=0)
