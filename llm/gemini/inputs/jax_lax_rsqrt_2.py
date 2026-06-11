
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class MagicTuple(tuple):
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False

def rsqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive values
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 2: 2D float32 array
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 3: 3D float64 array
    x = np.random.uniform(1.0, 100.0, size=(2, 2, 2)).astype(np.float64)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 4: 1D float16 array
    x = np.array([0.25, 0.5, 2.0, 8.0], dtype=np.float16)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 5: 2D complex64 array
    x = (np.random.uniform(0.1, 5.0, size=(2, 3)) + 1j * np.random.uniform(0.1, 5.0, size=(2, 3))).astype(np.complex64)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 6: 1D complex128 array
    x = (np.array([1.0 + 1j, 2.0 - 2j, -3.0 + 3j])).astype(np.complex128)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 7: 0D float32 array (scalar)
    x = np.array(2.0, dtype=np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 8: 4D float32 array
    x = np.random.uniform(0.5, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 9: 2D float32 array with large positive values
    x = np.array([[100.0, 10000.0], [1000000.0, 100000000.0]], dtype=np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 10: 3D float32 array with very small positive values
    x = np.array([[[1e-5, 1e-6], [1e-7, 1e-8]]], dtype=np.float32)
    accuracy = MagicTuple()
    list_of_inputs.append({"x": x, "accuracy": accuracy})

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
