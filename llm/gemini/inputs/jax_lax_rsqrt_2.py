
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rsqrt_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(4.0, dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D array float32
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D array float64
    x = np.array([[2.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D array float32
    x = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D array complex64
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, -2.0 - 5.0j], dtype=np.complex64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D array complex128
    x = np.array(
        [[-1.0 + 1.0j, 2.0 - 3.0j], [4.0 + 0.0j, -5.0 - 2.0j]],
        dtype=np.complex128,
    )
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D array float32
    x = np.random.uniform(0.5, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D array float16
    x = np.array([0.25, 1.0, 4.0, 16.0], dtype=np.float16)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 2D array float32 with negative values
    x = np.array([[-1.0, 2.0], [4.0, -9.0]], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 1D array float64 large values
    x = np.array([1e6, 1e8, 1e12], dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3D array complex64
    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(
        np.complex64
    )
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: 0D scalar complex128
    x = np.array(-4.0 + 3.0j, dtype=np.complex128)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.rsqrt_2"] = rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rsqrt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rsqrt_2'.")


check_valid('jax.lax.rsqrt', generated_inputs['jax.lax.rsqrt_2'], lib="jax", suffix=2)
