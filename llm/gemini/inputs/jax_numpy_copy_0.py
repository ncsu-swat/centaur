
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def copy_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 2: 2D array of floats
    a = np.random.randn(3, 4).astype(np.float32)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 3: 3D array of booleans
    a = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 4: 4D array of complex numbers
    a = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex64)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 5: 0D array (scalar)
    a = np.array(3.14, dtype=np.float32)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 6: 2D array of int16 zeros
    a = np.zeros((5, 5), dtype=np.int16)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 7: 1D array of float64
    a = np.array([-1.5, 0.0, 2.5, 9.9], dtype=np.float64)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 8: 5D array of uint8
    a = np.random.randint(0, 255, size=(2, 2, 2, 2, 2), dtype=np.uint8)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 9: 3D array of float16
    a = np.random.randn(2, 3, 4).astype(np.float16)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    # Input 10: 2D array of int64 with negative values
    a = np.random.randint(-100, 100, size=(4, 2), dtype=np.int64)
    order = "K"
    list_of_inputs.append({"a": a, "order": order})

    return list_of_inputs

generated_inputs["jax.numpy.copy"] = copy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.copy'.")


check_valid('jax.numpy.copy', generated_inputs['jax.numpy.copy'], lib="jax", suffix=0)
