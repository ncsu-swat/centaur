
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.lax.lax

_orig_tan = jax.lax.tan
def patched_tan(x, *, accuracy=None):
    return _orig_tan(x, accuracy=None)
jax.lax.tan = patched_tan
jax._src.lax.lax.tan = patched_tan

def tan_inputs():
    list_of_inputs = []

    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.random.randn(3, 4).astype(np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.random.randn(2, 2, 2).astype(np.float64)
    accuracy = (1e-7, 1e-7)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.array(0.5, dtype=np.float32)
    accuracy = (1e-6, 1e-6)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.random.uniform(-1.0, 1.0, (2, 3, 4, 5)).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.linspace(-10.0, 10.0, 10).astype(np.float64)
    accuracy = (1e-8, 1e-8)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = (np.random.randn(2, 2, 2) + 1j * np.random.randn(2, 2, 2)).astype(np.complex128)
    accuracy = (1e-9, 1e-9)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.random.randn(2, 1, 2, 1, 3).astype(np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    accuracy = (1e-6, 1e-6)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    return list_of_inputs

generated_inputs["jax.lax.tan_3"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_3'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_3'], lib="jax", suffix=3)
