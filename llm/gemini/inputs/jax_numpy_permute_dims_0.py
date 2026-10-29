
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def permute_dims_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar array
    a = np.array(42.0, dtype=np.float32)
    axes = ()
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 2: 1-D array
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 3: 2-D float array, standard transpose
    a = np.random.randn(3, 5).astype(np.float32)
    axes = (1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 4: 2-D boolean array, identity permutation
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    axes = (0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 5: 3-D int64 array, cyclic shift
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    axes = (1, 2, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 6: 3-D float64 array, reverse axes
    a = np.random.randn(4, 5, 6).astype(np.float64)
    axes = (2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 7: 4-D complex64 array
    a = (np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)).astype(
        np.complex64
    )
    axes = (3, 0, 2, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 8: 4-D uint8 array
    a = np.random.randint(0, 256, size=(1, 8, 16, 3), dtype=np.uint8)
    axes = (0, 3, 1, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 9: 5-D float32 array
    a = np.random.randn(2, 2, 3, 2, 4).astype(np.float32)
    axes = (4, 3, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    # Input 10: 6-D int16 array with singleton dimensions
    a = np.ones((1, 2, 1, 3, 1, 4), dtype=np.int16)
    axes = (5, 4, 3, 2, 1, 0)
    list_of_inputs.append(copy.deepcopy({"a": a, "axes": axes}))

    return list_of_inputs


generated_inputs["jax.numpy.permute_dims"] = permute_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.permute_dims' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.permute_dims'.")


check_valid('jax.numpy.permute_dims', generated_inputs['jax.numpy.permute_dims'], lib="jax", suffix=0)
