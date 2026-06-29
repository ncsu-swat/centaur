
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def diagonal_inputs():
    list_of_inputs = []

    # Input 1: 2D square matrix, offset=0, float32
    x = np.random.randn(3, 3).astype(np.float32)
    offset = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 2: 2D rectangular matrix, positive offset, int32
    x = np.random.randint(-10, 10, size=(4, 5)).astype(np.int32)
    offset = 1
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 3: 2D rectangular matrix, negative offset, float64
    x = np.random.randn(5, 4).astype(np.float64)
    offset = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 4: Large 2D square matrix, larger positive offset, float32
    x = np.random.randn(10, 10).astype(np.float32)
    offset = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 5: Batched 3D matrix, offset=0, int32
    x = np.random.randint(-100, 100, size=(2, 3, 3)).astype(np.int32)
    offset = 0
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 6: Batched 3D matrix, positive offset, float32
    x = np.random.randn(3, 4, 5).astype(np.float32)
    offset = 2
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 7: Batched 3D matrix, negative offset, float64
    x = np.random.randn(2, 5, 4).astype(np.float64)
    offset = -2
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 8: 4D batched matrix, negative offset, float32
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    offset = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 9: 5D batched matrix, complex64, positive offset
    x = (np.random.randn(1, 2, 3, 5, 5) + 1j * np.random.randn(1, 2, 3, 5, 5)).astype(np.complex64)
    offset = 3
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    # Input 10: 2D matrix, int64, negative offset
    x = np.random.randint(-5, 5, size=(2, 2)).astype(np.int64)
    offset = -1
    list_of_inputs.append({"x": copy.deepcopy(x), "offset": offset})

    return list_of_inputs

generated_inputs["jax.numpy.linalg.diagonal"] = diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.diagonal'.")


check_valid('jax.numpy.linalg.diagonal', generated_inputs['jax.numpy.linalg.diagonal'], lib="jax", suffix=0)
