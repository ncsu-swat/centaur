
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def silu_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array with mixed negative and positive values
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.random.randn(4, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 2D array float16
    x = np.random.uniform(-5.0, 5.0, size=(8, 8)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array (batch-like dimension)
    x = np.random.randn(2, 4, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 4D array (image-like dimension NCHW)
    x = np.random.randn(2, 3, 16, 16).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: All zeros array
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Extreme positive and negative values
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty array
    x = np.empty((0, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D tensor
    x = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array complex64
    x = np.array([1.0 + 2.0j, -1.0 - 0.5j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.nn.silu"] = silu_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.silu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.silu'.")


check_valid('jax.nn.silu', generated_inputs['jax.nn.silu'], lib="jax", suffix=0)
