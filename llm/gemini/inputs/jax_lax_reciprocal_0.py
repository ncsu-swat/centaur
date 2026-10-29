
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 1D array float32 with positive and negative values
    x = np.array([-4.0, -2.0, -0.5, 0.5, 2.0, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 2D array float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array float16
    x = np.array([0.1, 0.2, 0.5, 1.0, 10.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 3D array float32
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 1D array complex64
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 1.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 2D array complex128
    x = np.array(
        [[2.0 - 1.0j, 3.0 + 0.5j], [-1.0 + 2.0j, 4.0 - 4.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 4D array float32
    x = np.random.uniform(-10.0, -1.0, size=(2, 2, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large values float64
    x = np.array([1e5, 1e10, -1e8, 2.5e6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 5D array float32
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 5.0
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array small values float64
    x = np.array([1e-4, -2e-5, 3e-6, -4e-7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.lax.reciprocal"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reciprocal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reciprocal'.")


check_valid('jax.lax.reciprocal', generated_inputs['jax.lax.reciprocal'], lib="jax", suffix=0)
