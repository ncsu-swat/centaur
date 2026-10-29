
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive and negative values
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 2: 2D float64 array
    x = np.array([[-1.0, 0.25], [0.75, -0.25]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 3: 0D scalar float32 array
    x = np.array(0.42, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 4: 3D float16 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 5: 1D complex64 array
    x = np.array([0.5 + 0.5j, -0.5 + 1.0j, 1.0 - 0.5j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 6: 2D complex128 array
    x = np.array(
        [[0.1 - 0.2j, -0.3 + 0.4j], [0.5 + 0.0j, -0.6 - 0.7j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 7: 4D float32 array
    x = np.random.uniform(-1.2, 1.2, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 8: 1D float32 array containing zeros and small values
    x = np.array([0.0, -0.0, 1e-5, -1e-5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 9: 5D float32 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 10: 2D float32 array with negative and positive values
    x = np.array([[-0.785, 0.785], [-0.5, 0.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    # Input 11: 3D complex64 array
    x = (
        np.random.uniform(-1.0, 1.0, size=(2, 2, 3))
        + 1j * np.random.uniform(-1.0, 1.0, size=(2, 2, 3))
    ).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": None}))

    return list_of_inputs


generated_inputs["jax.lax.tan_1"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_1'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_1'], lib="jax", suffix=1)
