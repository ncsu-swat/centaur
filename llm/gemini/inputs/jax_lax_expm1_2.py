
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expm1_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 near zero
    x = np.array(1e-5, dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D array float32 with values around zero
    x = np.array([-1e-4, -1e-7, 0.0, 1e-7, 1e-4], dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 1D array float64 spanning negative to positive range
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 2D array float32
    x = np.random.randn(3, 4).astype(np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 2D array float64 with strictly negative values
    x = -np.random.exponential(scale=0.1, size=(2, 5)).astype(np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 3D array float16
    x = np.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(np.float16)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 3D array float32 with uniform negative values
    x = np.full((2, 2, 2), -1.5, dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 4D array float32
    x = np.random.randn(2, 1, 3, 2).astype(np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 1D complex64 array
    x = np.array([0.1 + 0.2j, -0.5 + 1.0j, 0.0 - 0.1j], dtype=np.complex64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D complex128 array
    x = np.array(
        [[1e-5 + 1e-5j, -2.0 + 0.5j], [0.0 + 0.0j, 1.0 - 1.0j]],
        dtype=np.complex128,
    )
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 1D float64 with very small values near zero
    x = np.array([-1e-15, -1e-10, 1e-10, 1e-15], dtype=np.float64)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: 2D float32 array of zeros
    x = np.zeros((4, 4), dtype=np.float32)
    accuracy = ()
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.expm1_2"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expm1_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expm1_2'.")


check_valid('jax.lax.expm1', generated_inputs['jax.lax.expm1_2'], lib="jax", suffix=2)
