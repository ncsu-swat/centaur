
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def transpose_inputs():
    list_of_inputs = []

    # Input 1: 1D array, identity transpose
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 2: 2D array, standard matrix transpose
    a = np.random.randn(2, 3).astype(np.float32)
    axes = (1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 3: 2D array, identity permutation
    a = np.arange(12, dtype=np.int64).reshape((3, 4))
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 4: 3D array, full reversal
    a = np.random.randn(3, 4, 5).astype(np.float64)
    axes = (2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 5: 3D array, cyclic permutation
    a = np.arange(24, dtype=np.float32).reshape((2, 3, 4)) - 10.0
    axes = (1, 2, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 6: 4D array, float32, arbitrary permutation
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axes = (0, 3, 1, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 7: 4D array, boolean
    a = np.array([True, False] * 12, dtype=bool).reshape((2, 2, 2, 3))
    axes = (3, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 8: 5D array, complex numbers
    a = (np.random.randn(1, 2, 3, 2, 1) + 1j * np.random.randn(1, 2, 3, 2, 1)).astype(
        np.complex64
    )
    axes = (4, 3, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 9: 0D scalar array
    a = np.array(42.0, dtype=np.float32)
    axes = ()
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 10: 3D array with negative numbers and uint8 dtype
    a = np.random.randint(0, 256, size=(4, 2, 3), dtype=np.uint8)
    axes = (2, 0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 11: 6D array with singleton dimensions
    a = np.zeros((1, 2, 1, 3, 1, 4), dtype=np.int16)
    axes = (5, 4, 3, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

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
