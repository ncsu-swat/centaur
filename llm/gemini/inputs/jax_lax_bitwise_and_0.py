
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_and_inputs():
    list_of_inputs = []

    # Input 1: Boolean 1D arrays
    x = np.array([True, False, True, False], dtype=bool)
    y = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: Int32 1D arrays
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([5, 6, 7, 8], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Int16 2D arrays with negative values
    x = np.array([[-1, 2], [-3, 4]], dtype=np.int16)
    y = np.array([[5, -6], [7, -8]], dtype=np.int16)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Int8 3D arrays, broadcasting with matching number of dimensions
    x = np.array([[[1], [2]], [[3], [4]]], dtype=np.int8)  # shape (2, 2, 1)
    y = np.array([[[5, 6]], [[7, 8]]], dtype=np.int8)      # shape (2, 1, 2)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Int64 1D arrays
    x = np.array([100000, 200000], dtype=np.int64)
    y = np.array([300000, 400000], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Int8 2D arrays
    x = np.array([[127, 0], [-128, 64]], dtype=np.int8)
    y = np.array([[15, 15], [-15, -15]], dtype=np.int8)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Boolean 3D arrays
    x = np.random.choice([True, False], size=(2, 2, 2)).astype(bool)
    y = np.random.choice([True, False], size=(2, 2, 2)).astype(bool)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Int32 2D arrays
    x = np.array([[1024, 2048], [4096, 8192]], dtype=np.int32)
    y = np.array([[1024, 0], [4096, 0]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Int64 2D arrays
    x = np.array([[4294967295, -1], [0, 1]], dtype=np.int64)
    y = np.array([[4294967295, 4294967295], [1, 0]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Int16 4D arrays, Broadcasting with size-1 dimensions
    x = np.ones((2, 1, 3, 1), dtype=np.int16) * 7
    y = np.ones((2, 2, 1, 5), dtype=np.int16) * 3
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Int8 1D arrays, broadcasting 1 element array to 3 elements
    x = np.array([15], dtype=np.int8)
    y = np.array([7, 8, 9], dtype=np.int8)
    list_of_inputs.append({"x": x, "y": y})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.lax.bitwise_and"] = bitwise_and_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_and' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_and'.")


check_valid('jax.lax.bitwise_and', generated_inputs['jax.lax.bitwise_and'], lib="jax", suffix=0)
