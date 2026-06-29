
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tensorsolve_inputs():
    list_of_inputs = []

    # Input 1
    a = np.random.randn(2, 2, 4).astype(np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    axes = (2,)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.random.randn(4, 2, 2).astype(np.float64)
    b = np.random.randn(2, 2).astype(np.float64)
    axes = (0,)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(3, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    axes = (1,)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(2, 3, 1, 6).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    axes = (2, 3)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.random.randn(2, 1, 6, 3).astype(np.float64)
    b = np.random.randn(2, 3).astype(np.float64)
    axes = (1, 2)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    axes = (2, 3)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.random.randn(1, 4, 1, 4).astype(np.float32)
    b = np.random.randn(1, 4).astype(np.float32)
    axes = (2, 3)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.random.randn(3, 1, 3).astype(np.float64)
    b = np.random.randn(3).astype(np.float64)
    axes = (1, 2)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.randn(2, 5, 2, 5).astype(np.float32)
    b = np.random.randn(2, 5).astype(np.float32)
    axes = (2, 3)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.random.randn(8, 8).astype(np.float64)
    b = np.random.randn(8).astype(np.float64)
    axes = (1,)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    a = np.random.randn(2, 2, 2, 2, 16).astype(np.float32)
    b = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axes = (4,)
    input_dict = {"a": a, "b": b, "axes": axes}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.tensorsolve"] = tensorsolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.tensorsolve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.tensorsolve'.")


check_valid('jax.numpy.linalg.tensorsolve', generated_inputs['jax.numpy.linalg.tensorsolve'], lib="jax", suffix=0)
