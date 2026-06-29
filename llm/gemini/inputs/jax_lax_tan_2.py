
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 2: 2D float32 array with negatives
    x = np.random.uniform(-1.5, 1.5, size=(3, 3)).astype(np.float32)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 3: 3D float64 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float64)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 4: 0D scalar float32
    x = np.array(0.75, dtype=np.float32)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 5: 1D complex64 array
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex64)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 6: 2D complex128 array
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex128)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 7: 4D float32 array
    x = np.random.uniform(-0.5, 0.5, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 8: 1D float64 array
    x = np.array([-0.785398, 0.0, 0.785398], dtype=np.float64)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 9: 3D float32 array
    x = np.random.uniform(-2.0, 2.0, size=(3, 1, 4)).astype(np.float32)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 10: 2D float64 array with larger dimension
    x = np.random.uniform(-1.2, 1.2, size=(10, 10)).astype(np.float64)
    accuracy = None
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    return list_of_inputs

generated_inputs["jax.lax.tan_2"] = jax_lax_tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_2'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_2'], lib="jax", suffix=2)
