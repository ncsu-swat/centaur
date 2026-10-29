
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax
import numpy as np


def log1p_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 near zero
    x = np.array(1e-5, dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D array float32 with negative values > -1
    x = np.array([-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D array float64 with very small values
    x = np.array(
        [[1e-15, -1e-15], [1e-10, -1e-10], [1e-5, -1e-5]], dtype=np.float64
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D array float32 with shape (2, 3, 4)
    x = np.random.uniform(low=-0.8, high=5.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D array float16
    x = np.array([-0.75, -0.25, 0.0, 0.25, 0.75, 2.0], dtype=np.float16)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D array complex64
    x = np.array(
        [[0.1 + 0.2j, -0.3 + 0.4j], [0.5 - 0.1j, -0.8 + 0.0j]],
        dtype=np.complex64,
    )
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 1D array complex128
    x = np.array(
        [-0.5 + 1.0j, 0.0 + 0.0j, 1.5 - 2.0j, -0.99 + 0.01j],
        dtype=np.complex128,
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 4D array float32
    x = np.random.uniform(low=0.0, high=10.0, size=(2, 2, 3, 3)).astype(
        np.float32
    )
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 2D array float64 strictly negative in (-0.99, -0.01)
    x = np.linspace(-0.99, -0.01, num=12, dtype=np.float64).reshape((3, 4))
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 1D array float32 with large positive values
    x = np.array([10.0, 100.0, 1000.0, 10000.0, 1e6], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3D array complex64
    real_part = np.random.uniform(-0.5, 0.5, size=(2, 2, 2)).astype(np.float32)
    imag_part = np.random.uniform(-0.5, 0.5, size=(2, 2, 2)).astype(np.float32)
    x = real_part + 1j * imag_part
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.log1p"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log1p' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log1p'.")


check_valid('jax.lax.log1p', generated_inputs['jax.lax.log1p'], lib="jax", suffix=0)
