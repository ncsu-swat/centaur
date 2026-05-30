
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, swapping axes
    a = np.random.randn(3, 4).astype(np.float32)
    axes = (1, 0)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 2: 2D int32 array, identity transpose
    a = np.random.randint(-10, 10, size=(5, 5)).astype(np.int32)
    axes = (0, 1)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 3: 3D float64 array, permutation (2, 0, 1)
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axes = (2, 0, 1)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 4: 3D uint8 array, permutation (1, 2, 0)
    a = np.random.randint(0, 255, size=(4, 2, 3)).astype(np.uint8)
    axes = (1, 2, 0)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 5: 4D float32 array, reversing all axes
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = (3, 2, 1, 0)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 6: 4D int16 array, swapping middle axes
    a = np.random.randint(-100, 100, size=(2, 5, 5, 3)).astype(np.int16)
    axes = (0, 2, 1, 3)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 7: 5D float32 array, complex permutation
    a = np.random.randn(2, 2, 3, 4, 2).astype(np.float32)
    axes = (4, 0, 3, 1, 2)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 8: 1D bool array, trivial transpose
    a = np.random.choice([True, False], size=(10,)).astype(np.bool_)
    axes = (0,)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 9: 3D complex64 array, swapping first two axes
    a = (np.random.randn(3, 3, 2) + 1j * np.random.randn(3, 3, 2)).astype(np.complex64)
    axes = (1, 0, 2)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 10: 2D bool array, swapping axes
    a = np.random.choice([True, False], size=(4, 5)).astype(np.bool_)
    axes = (1, 0)
    list_of_inputs.append({"a": a, "axes": axes})

    # Input 11: 6D float32 array, reversing all axes
    a = np.random.randn(2, 2, 2, 2, 2, 2).astype(np.float32)
    axes = (5, 4, 3, 2, 1, 0)
    list_of_inputs.append({"a": a, "axes": axes})

    return list_of_inputs

generated_inputs["jax.numpy.transpose_1"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.transpose_1'.")


check_valid('jax.numpy.transpose', generated_inputs['jax.numpy.transpose_1'], lib="jax", suffix=1)
