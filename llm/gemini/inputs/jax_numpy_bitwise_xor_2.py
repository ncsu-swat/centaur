
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive int
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 array, negative int
    x = np.array([[-5, 10], [15, -20]], dtype=np.int64)
    y = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 3D int32 array, zero
    x = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    y = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalar int32 array, positive int
    x = np.array(42, dtype=np.int32)
    y = 255
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 1D int64 array, positive int
    x = np.array([0, 127, 255, 1024, 65535], dtype=np.int64)
    y = 15
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D int32 array, positive int
    x = np.random.randint(-100, 100, size=(1, 2, 3, 2), dtype=np.int32)
    y = 7
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 2D int64 array, large int
    x = np.array([[1000, 2000], [3000, 4000]], dtype=np.int64)
    y = 1024
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 1D int32 array with powers of 2
    x = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.int32)
    y = 4096
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int32 array with negatives, negative int
    x = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    y = -8
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 5D int32 array, positive int
    x = np.ones((2, 1, 2, 1, 3), dtype=np.int32)
    y = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D int64 array with zeros
    x = np.zeros((2, 3, 2), dtype=np.int64)
    y = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_xor_2"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_2'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_2'], lib="jax", suffix=2)
