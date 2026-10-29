
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def tanh_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float32 matrix
    x = np.array(
        [[-1.5, -0.5, 0.0], [0.5, 1.5, 2.0]],
        dtype=np.float32,
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D scalar float32
    x = np.array(0.75, dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 1D float64 array with wide dynamic range
    x = np.array([-10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 2D float64 array
    x = np.linspace(-3.0, 3.0, 12, dtype=np.float64).reshape(3, 4)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 3D float16 array
    x = np.array(
        [[[-1.0, 0.0], [1.0, 2.0]], [[-2.0, -0.5], [0.5, 1.5]]],
        dtype=np.float16,
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 1D complex64 array
    x = np.array(
        [0.0 + 0.0j, 1.0 + 1.0j, -1.0 - 1.0j, 0.5 - 0.5j], dtype=np.complex64
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 2D complex128 array
    x = np.array(
        [[-0.5 + 1.0j, 0.0 - 0.5j], [1.5 + 0.0j, -1.0 - 2.0j]],
        dtype=np.complex128,
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 4D float32 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: Large positive and negative values (saturation regions)
    x = np.array([-100.0, -50.0, -20.0, 20.0, 50.0, 100.0], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 5D float32 array
    x = np.ones((1, 2, 1, 3, 2), dtype=np.float32) * -0.5
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: Very small values near zero
    x = np.array([-1e-7, -1e-15, 0.0, 1e-15, 1e-7], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.tanh"] = tanh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tanh'.")


check_valid('jax.lax.tanh', generated_inputs['jax.lax.tanh'], lib="jax", suffix=0)
