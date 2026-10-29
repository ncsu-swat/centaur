
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def log_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32, positive value, default accuracy mode
    x = np.array(2.718281828, dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D float32 array, strictly positive values, highest accuracy mode
    x = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D float64 array, positive values, default accuracy mode
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float16 array, small positive values, default accuracy mode
    x = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float16)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D complex64 array, highest accuracy mode
    x = np.array([1.0 + 1.0j, -2.0 + 0.5j, 0.0 + 3.0j], dtype=np.complex64)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex128 array, default accuracy mode
    x = np.array(
        [[1.0 - 2.0j, -3.0 - 4.0j], [0.5 + 0.5j, 2.0 + 0.0j]],
        dtype=np.complex128,
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D float32 array, random positive values, default accuracy mode
    x = np.random.uniform(1.0, 100.0, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D float64 array with zero, infinity, and large values, default accuracy mode
    x = np.array([0.0, 1e-10, 1e10, np.inf], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 3D float32 array with fractions, highest accuracy mode
    x = np.array([[[0.125, 0.25]], [[0.5, 0.75]]], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.HIGHEST
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D float32 array with powers of e, default accuracy mode
    x = np.exp(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.log_1"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log_1'.")


check_valid('jax.lax.log', generated_inputs['jax.lax.log_1'], lib="jax", suffix=1)
