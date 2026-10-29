
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean arrays, same shape
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([True, True, False, False], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 1D int32 arrays with positive values
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = np.array([1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D int64 arrays with negative and positive values
    x = np.array([[-1, 2], [3, -4]], dtype=np.int64)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: Broadcasting 2D with 1D int32
    x = np.array([[0, 255], [128, 64]], dtype=np.int32)
    y = np.array([1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalars as numpy arrays (scalar tensors)
    x = np.array(7, dtype=np.int32)
    y = np.array(12, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Broadcasting with different dimensions (3D and 1D)
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    y = np.array([1, 2, 4, 8], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D int64 arrays with broadcasting
    x = np.zeros((1, 2, 1, 4), dtype=np.int64)
    y = np.ones((3, 1, 5, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 1D int32 arrays with negative and edge values
    x = np.array([-128, 127, 0, -1], dtype=np.int32)
    y = np.array([127, -128, -1, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 2D boolean arrays with broadcasting
    x = np.array([[True], [False], [True]], dtype=np.bool_)
    y = np.array([[False, True, False]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: Empty arrays
    x = np.array([], dtype=np.int32)
    y = np.array([], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.bitwise_or_1"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_1'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_1'], lib="jax", suffix=1)
