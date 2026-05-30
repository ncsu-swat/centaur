
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def permute_dims_inputs():
    list_of_inputs = []

    # Input 1: 2D float32, transpose axes
    a = np.random.randn(3, 5).astype(np.float32)
    axes = (1, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int32, axes permutation (0, 2, 1)
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    axes = (0, 2, 1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float64, cyclic permutation (2, 0, 1)
    a = np.random.randn(4, 5, 6).astype(np.float64)
    axes = (2, 0, 1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D uint8, complete reverse (3, 2, 1, 0)
    a = np.random.randint(0, 256, size=(2, 2, 3, 3)).astype(np.uint8)
    axes = (3, 2, 1, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D bool, identity permute (0,)
    a = np.random.choice([True, False], size=(10,)).astype(bool)
    axes = (0,)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D float32, complex permute (4, 0, 2, 1, 3)
    a = np.random.randn(2, 3, 2, 4, 2).astype(np.float32)
    axes = (4, 0, 2, 1, 3)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D complex64, identity (0, 1)
    a = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    axes = (0, 1)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int16, axes permutation (1, 2, 0)
    a = np.random.randint(-100, 100, size=(3, 4, 5)).astype(np.int16)
    axes = (1, 2, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float16, pairwise swap (1, 0, 3, 2)
    a = np.random.randn(2, 2, 4, 4).astype(np.float16)
    axes = (1, 0, 3, 2)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6D int64, complete reverse (5, 4, 3, 2, 1, 0)
    a = np.random.randint(-1000, 1000, size=(2, 2, 2, 2, 2, 2)).astype(np.int64)
    axes = (5, 4, 3, 2, 1, 0)
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
