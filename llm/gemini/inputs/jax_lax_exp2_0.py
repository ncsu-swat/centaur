
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


def exp2_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(2.0, dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D float32 array with negative and positive values
    x = np.array([-3.0, -1.0, 0.0, 1.5, 4.0], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D float64 matrix
    x = np.array([[-2.0, -0.5, 1.0], [0.0, 2.5, 4.0]], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 1D float16 array
    x = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float16)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 3D float32 tensor
    x = np.random.randn(2, 3, 4).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 1D complex64 array
    x = np.array(
        [1.0 + 2.0j, -0.5 - 1.5j, 0.0 + 0.0j, 2.0 - 1.0j], dtype=np.complex64
    )
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 2D complex128 matrix
    real_part = np.array([[-1.0, 2.0], [0.5, -2.5]], dtype=np.float64)
    imag_part = np.array([[0.5, -1.0], [2.0, 1.5]], dtype=np.float64)
    x = (real_part + 1j * imag_part).astype(np.complex128)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 4D float32 tensor
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 1D float64 array with wide exponent range
    x = np.array([-20.0, -5.0, 0.0, 5.0, 20.0], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D empty float32 array
    x = np.empty((0, 4), dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 3D complex64 tensor
    x = (
        np.random.randn(2, 3, 2).astype(np.float32)
        + 1j * np.random.randn(2, 3, 2).astype(np.float32)
    ).astype(np.complex64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: 5D float32 tensor
    x = np.random.randn(1, 2, 2, 2, 3).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.exp2"] = exp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.exp2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.exp2'.")


check_valid('jax.lax.exp2', generated_inputs['jax.lax.exp2'], lib="jax", suffix=0)
