
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D array float32, standard swap
    a = np.random.randn(3, 4).astype(np.float32)
    axes = [1, 0]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array float64, permutation
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axes = [2, 0, 1]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array int32, identity transpose
    a = np.arange(10).astype(np.int32)
    axes = [0]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array int64, permutation
    a = np.random.randint(0, 10, size=(2, 2, 3, 3)).astype(np.int64)
    axes = [0, 2, 1, 3]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array uint8, shift axes
    a = np.random.randint(0, 256, size=(5, 5, 3)).astype(np.uint8)
    axes = [2, 0, 1]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 5D array float32, complete reversal
    a = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    axes = [4, 3, 2, 1, 0]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array boolean, keep same order
    a = (np.random.randn(5, 5) > 0).astype(bool)
    axes = [0, 1]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array complex64, permutation
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    axes = [1, 2, 0]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array float16, permutation
    a = np.random.randn(2, 4, 2, 4).astype(np.float16)
    axes = [1, 3, 0, 2]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array int16, standard swap
    a = np.random.randint(-100, 100, size=(10, 5)).astype(np.int16)
    axes = [1, 0]
    input_dict = {"a": a, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.transpose_2"] = transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.transpose_2'.")


check_valid('jax.numpy.transpose', generated_inputs['jax.numpy.transpose_2'], lib="jax", suffix=2)
