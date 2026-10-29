
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asin_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D float32 array spanning [-1.0, 1.0]
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D float64 array
    x = np.random.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D float16 array
    x = np.array([-0.8, -0.2, 0.0, 0.2, 0.8], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D float32 array
    x = np.random.uniform(-0.9, 0.9, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D complex64 array
    x = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 1D complex128 array
    x = np.array(
        [1.0 + 2.0j, -0.5 - 0.5j, 0.0 + 0.0j, -2.0 + 1.0j],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(1, 2, 2, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 2D float32 boundary values
    x = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 3D complex64 array with zeros
    x = np.zeros((2, 1, 3), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.asin'.")


check_valid('jax.lax.asin', generated_inputs['jax.lax.asin'], lib="jax", suffix=0)
