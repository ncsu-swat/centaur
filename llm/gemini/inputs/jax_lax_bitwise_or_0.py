
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: Boolean arrays, 1D, same shape
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, True, True, False], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: Int32 arrays, 1D, same shape
    x = np.array([1, 2, 4, 8], dtype=np.int32)
    y = np.array([8, 4, 2, 1], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: Int32 arrays with negative values, 2D, same shape
    x = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    y = np.array([[5, -6], [-7, 8]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: Int64 arrays, 3D, same shape
    x = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64)
    y = np.random.randint(-100, 100, size=(2, 3, 4)).astype(np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: Int64 arrays, broadcasting (1, 3) and (3, 3)
    x = np.array([[1, 2, 3]], dtype=np.int64)
    y = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: Boolean arrays, broadcasting (2, 1) and (2, 2)
    x = np.array([[True], [False]], dtype=np.bool_)
    y = np.array([[False, True], [True, False]], dtype=np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Int32 arrays, negative values, 1D
    x = np.array([-128, 127, 0, -1], dtype=np.int32)
    y = np.array([127, -128, -1, 0], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: Int64 arrays, 2D, same shape
    x = np.array([[123456789, 987654321], [0, -1]], dtype=np.int64)
    y = np.array([[987654321, 123456789], [-1, 0]], dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Boolean arrays, 4D, same shape
    x = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    y = np.random.choice([True, False], size=(2, 2, 2, 2)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: Int32 arrays, broadcasting (3, 1) and (1, 3)
    x = np.array([[1], [2], [3]], dtype=np.int32)
    y = np.array([[4, 5, 6]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 11: Boolean arrays, 3D, same shape
    x = np.random.choice([True, False], size=(3, 3, 3)).astype(np.bool_)
    y = np.random.choice([True, False], size=(3, 3, 3)).astype(np.bool_)
    list_of_inputs.append({"x": x, "y": y})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

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
