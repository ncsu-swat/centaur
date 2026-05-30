
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def choose_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D index, 2D choices with 'wrap' mode
    a = np.array([0, 1, 2, 0], dtype=np.int32)
    choices = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]], dtype=np.int32)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    # Input 2: Out of bounds indices with 'clip' mode
    a = np.array([3, -1, 1, 2], dtype=np.int32)
    choices = np.array([[10, 20, 30, 40],
                        [50, 60, 70, 80],
                        [90, 100, 110, 120]], dtype=np.int32)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "clip"})

    # Input 3: Float32 choice arrays
    a = np.array([1, 0, 1], dtype=np.int32)
    choices = np.array([[1.5, 2.5, 3.5],
                        [4.5, 5.5, 6.5]], dtype=np.float32)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    # Input 4: 2D index array and 3D choices
    a = np.array([[0, 1], [1, 0]], dtype=np.int32)
    choices = np.array([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]], dtype=np.int32)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    # Input 5: Broadcast dimension for index array
    a = np.array([[0], [1], [2]], dtype=np.int32)  # shape (3, 1)
    choices = np.array([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10], [11, 12]],
                        [[13, 14], [15, 16], [17, 18]]], dtype=np.int32)  # slices of shape (3, 2)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "clip"})

    # Input 6: Large integer choices, 1D indexing
    a = np.random.randint(-5, 10, size=(10,), dtype=np.int64)
    choices = np.random.randint(0, 100, size=(4, 10), dtype=np.int64)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    # Input 7: Float64 choices with 2D index array
    a = np.random.randint(0, 5, size=(5, 5), dtype=np.int32)
    choices = np.random.randn(5, 5, 5).astype(np.float64)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "clip"})

    # Input 8: Int16 indexing
    a = np.array([0, 1, 0, 1], dtype=np.int16)
    choices = np.array([[100, 200, 300, 400], [500, 600, 700, 800]], dtype=np.int16)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    # Input 9: 3D choices with broadcasting 1D index
    a = np.array([0, 1], dtype=np.int32)  # shape (2,)
    choices = np.random.rand(2, 3, 2).astype(np.float32)  # slices of shape (3, 2)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "clip"})

    # Input 10: Large out-of-bound indices with 'wrap'
    a = np.array([-10, 10], dtype=np.int32)
    choices = np.random.rand(3, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "choices": choices, "mode": "wrap"})

    return list_of_inputs

generated_inputs["jax.numpy.choose_2"] = choose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.choose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.choose_2'.")


check_valid('jax.numpy.choose', generated_inputs['jax.numpy.choose_2'], lib="jax", suffix=2)
