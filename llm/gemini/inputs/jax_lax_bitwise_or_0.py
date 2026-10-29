
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x = np.array([1, 2, 4, 8], dtype=np.int32)
    y = np.array([8, 4, 2, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int64 arrays with negative values and broadcasting
    x = np.array([[-1], [0], [15]], dtype=np.int64)
    y = np.array([[1, -2, 3, -4]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 1D boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 2D int32 arrays
    x = np.array([[255, 0], [128, 64]], dtype=np.int32)
    y = np.array([[0, 255], [64, 128]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 3D int32 arrays
    x = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    y = np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 0D scalar int32 and 2D int32 array
    x = np.array(7, dtype=np.int32)
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: 0D boolean arrays
    x = np.array(True, dtype=np.bool_)
    y = np.array(False, dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 4D int32 arrays with broadcasting
    x = np.zeros((2, 1, 3, 1), dtype=np.int32)
    y = np.ones((1, 4, 1, 5), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 2D int64 arrays with negative values
    x = np.array([[-128, 127], [-1, 0]], dtype=np.int64)
    y = np.array([[127, -128], [0, -1]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 3D boolean arrays
    x = np.random.choice([True, False], size=(2, 2, 3)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 2, 3)).astype(np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.bitwise_or"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_or' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_or'.")


check_valid('jax.lax.bitwise_or', generated_inputs['jax.lax.bitwise_or'], lib="jax", suffix=0)
