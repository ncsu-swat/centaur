
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.lax

_orig_log = jax.lax.log
def _patched_log(x, *, accuracy=None):
    return _orig_log(x, accuracy=None)
jax.lax.log = _patched_log

def log_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 positive array, accuracy tuple
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 2: 2D float32 positive array, accuracy tuple
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 3: 3D float64 positive array, accuracy tuple
    x = np.random.uniform(0.1, 100.0, size=(2, 2, 2)).astype(np.float64)
    accuracy = (1e-7, 1e-7)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 4: 1D float16 positive array, accuracy tuple
    x = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    accuracy = (1e-2, 1e-2)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 5: 2D complex64 array, accuracy tuple
    x = (np.random.uniform(0.1, 5.0, size=(2, 3)) + 1j * np.random.uniform(0.1, 5.0, size=(2, 3))).astype(np.complex64)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 6: 1D complex128 array, accuracy tuple
    x = np.array([1.0 + 1j, 2.0 - 2j, 0.5 + 0.5j], dtype=np.complex128)
    accuracy = (1e-6, 1e-6)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 7: 0D (scalar-like) float32 array, accuracy tuple
    x = np.array(5.0, dtype=np.float32)
    accuracy = (1e-3, 1e-3)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 8: 4D float32 positive array, accuracy tuple
    x = np.random.uniform(1.0, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 9: 2D float64 positive array with large values, accuracy tuple
    x = np.random.uniform(10.0, 1000.0, size=(4, 2)).astype(np.float64)
    accuracy = (1e-8, 1e-8)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 10: 3D complex64 array, accuracy tuple
    x = (np.random.uniform(1.0, 10.0, size=(2, 1, 3)) + 1j * np.random.uniform(1.0, 10.0, size=(2, 1, 3))).astype(np.complex64)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    return list_of_inputs

generated_inputs["jax.lax.log_2"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log_2'.")


check_valid('jax.lax.log', generated_inputs['jax.lax.log_2'], lib="jax", suffix=2)
