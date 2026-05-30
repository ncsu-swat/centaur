
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: Python integers (scalars)
    x = 5
    y = 5
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python integers with negative values
    x = -10
    y = 10
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D numpy integer arrays (int32)
    x = np.array([1, 2, -3], dtype=np.int32)
    y = np.array([1, -2, -3], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D numpy integer arrays (int64)
    x = np.random.randint(-100, 100, size=(5, 5), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(5, 5), dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting 1D to 2D (int16)
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int16)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unsigned integer array and unsigned scalar
    x = np.array([0, 100, 255], dtype=np.uint8)
    y = np.uint8(100)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Comparison with zeros (int32)
    x = np.arange(-5, 5, dtype=np.int32)
    y = np.zeros(10, dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D numpy integer arrays (int64)
    x = np.ones((2, 2, 2), dtype=np.int64)
    y = np.zeros((2, 2, 2), dtype=np.int64)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcasting single element 2D array to 1D array
    x = np.array([[5]], dtype=np.int32)
    y = np.array([5, 5, 5], dtype=np.int32)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Numpy integer scalars (int8)
    x = np.int8(-128)
    y = np.int8(127)
    input_dict = {"x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.equal_7"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_7'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_7'], lib="jax", suffix=7)
