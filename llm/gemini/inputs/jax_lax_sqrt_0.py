
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 2: 2D float64 matrix with random positive values
    x = np.random.uniform(0.1, 100.0, size=(4, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 3: 3D float32 tensor
    x = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 4: Scalar float32 input
    x = np.array(49.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 5: 1D complex64 array
    x = np.array(
        [1.0 + 2.0j, -4.0 + 0.0j, 0.0 + 3.0j, 9.0 - 4.0j], dtype=np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 6: 2D complex128 array
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 7: 4D float32 tensor
    x = np.random.uniform(0.0, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 8: 1D float16 array
    x = np.array([0.25, 0.5, 1.0, 2.0, 4.0], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 9: 2D float32 array using linspace
    x = np.linspace(0.0, 100.0, num=12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 10: 3D float64 array with large values
    x = np.random.uniform(1e6, 1e8, size=(2, 4, 2)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 11: 1D float32 array with small values close to zero
    x = np.array([1e-10, 1e-8, 1e-6, 1e-4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 12: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * 81.0
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

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
