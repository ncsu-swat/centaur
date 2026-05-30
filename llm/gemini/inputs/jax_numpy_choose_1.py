
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def choose_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array indices, choices as a 2D array
    a = np.array([2, 0, 1, 0], dtype=np.int32)
    choices = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype=np.int32)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Out of bound indices with 'wrap'
    a = np.array([2, 0, 1, 4], dtype=np.int32)
    choices = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype=np.int32)
    input_dict = {"a": a, "choices": choices, "mode": "wrap"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative indices with 'wrap'
    a = np.array([-1, -2, 0], dtype=np.int32)
    choices = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    input_dict = {"a": a, "choices": choices, "mode": "wrap"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of indices, 3D choices
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    choices = np.array([
        [[10, 20], [30, 40]],
        [[50, 60], [70, 80]]
    ], dtype=np.int32)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted choices, shape (2, 2, 3)
    a = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    choices = np.arange(12).reshape(2, 2, 3).astype(np.int32)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float choices (float64) with 1D index
    a = np.array([0, 1, 1], dtype=np.int64)
    choices = np.array([
        [1.5, 2.5, 3.5],
        [4.5, 5.5, 6.5]
    ], dtype=np.float64)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D index array
    a = np.random.randint(0, 3, size=(2, 2, 2)).astype(np.int32)
    choices = np.random.randn(3, 2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D arrays
    a = np.random.randint(-1, 4, size=(100,)).astype(np.int32)
    choices = np.random.randn(3, 100).astype(np.float32)
    input_dict = {"a": a, "choices": choices, "mode": "wrap"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element indices and choices
    a = np.array([0], dtype=np.int32)
    choices = np.array([[10], [20]], dtype=np.int32)
    input_dict = {"a": a, "choices": choices, "mode": "clip"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D index and choices
    a = np.random.randint(0, 2, size=(2, 2, 2, 2)).astype(np.int32)
    choices = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {"a": a, "choices": choices, "mode": "wrap"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.choose_1"] = choose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.choose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.choose_1'.")


check_valid('jax.numpy.choose', generated_inputs['jax.numpy.choose_1'], lib="jax", suffix=1)
