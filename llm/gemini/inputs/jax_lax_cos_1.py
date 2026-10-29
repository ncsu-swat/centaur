
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np

def cos_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalar array
    x = np.array(0.0, dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 1D float32 array with standard angle values
    x = np.array([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 2D float64 matrix
    x = np.random.randn(4, 5).astype(np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float32 tensor
    x = np.random.randn(2, 3, 4).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D complex64 array
    x = (np.random.randn(6) + 1j * np.random.randn(6)).astype(np.complex64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex128 matrix
    x = (np.random.randn(3, 4) + 1j * np.random.randn(3, 4)).astype(np.complex128)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D float16 tensor
    x = np.random.randn(2, 2, 3, 2).astype(np.float16)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D empty float32 array
    x = np.empty((0,), dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 5D float32 tensor
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D float32 array with special values
    x = np.array([[0.0, -0.0, np.nan], [np.inf, -np.inf, 1e5]], dtype=np.float32)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 1D float64 array with extreme/small values
    x = np.array([-1e-7, 1e-7, -100.0, 100.0], dtype=np.float64)
    accuracy = jax.lax.AccuracyMode.DEFAULT
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs

generated_inputs["jax.lax.cos_1"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cos_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cos_1'.")


check_valid('jax.lax.cos', generated_inputs['jax.lax.cos_1'], lib="jax", suffix=1)
