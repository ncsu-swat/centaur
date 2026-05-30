
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_right_shift_inputs():
    list_of_inputs = []

    # Input 1: Basic int32, 1D arrays
    x1 = np.array([16, 32, 64, 128], dtype=np.int32)
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 2: int32, 2D arrays
    x1 = np.array([[8, 16], [24, 32]], dtype=np.int32)
    x2 = np.array([[1, 2], [1, 2]], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 3: Negative signed integers (int32), 1D
    x1 = np.array([-16, -32, -64], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 4: Broadcasting, 2D x1 and 1D x2
    x1 = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 5: 0D arrays (scalars as tensors), int64
    x1 = np.array(1024, dtype=np.int64)
    x2 = np.array(5, dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 6: Large values with int64, 3D
    x1 = np.random.randint(1000, 100000, size=(2, 2, 2), dtype=np.int64)
    x2 = np.random.randint(1, 10, size=(2, 2, 2), dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 7: Broadcasting, 1D x1 and 3D x2, int32
    x1 = np.array([100], dtype=np.int32)
    x2 = np.random.randint(1, 5, size=(2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 8: int64, 2D arrays
    x1 = np.array([[1024, 2048], [4096, 8192]], dtype=np.int64)
    x2 = np.array([[2, 3], [4, 5]], dtype=np.int64)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 9: Large dimensions (4D), int32
    x1 = np.random.randint(0, 255, size=(2, 2, 2, 2), dtype=np.int32)
    x2 = np.random.randint(1, 4, size=(2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    # Input 10: Zero shifts, 1D array, int32
    x1 = np.array([5, 10, 15], dtype=np.int32)
    x2 = np.array([0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"x1": copy.deepcopy(x1), "x2": copy.deepcopy(x2)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_right_shift"] = bitwise_right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_right_shift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_right_shift'.")


check_valid('jax.numpy.bitwise_right_shift', generated_inputs['jax.numpy.bitwise_right_shift'], lib="jax", suffix=0)
