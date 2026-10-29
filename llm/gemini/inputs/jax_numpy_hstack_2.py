
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def hstack_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, shape (3,), float32
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays with negative values, shape (4,), int32
    a = np.array([-1, 0, 1, 2], dtype=np.int32)
    b = np.array([2, -3, 4, 5], dtype=np.int32)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, shape (2, 3), float32
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays with negative values, shape (3, 2), float64
    a = np.array([[-2.5, 1.0], [1.5, -0.5], [-0.5, 3.2]], dtype=np.float64)
    b = np.array([[3.0, -1.0], [0.0, 2.2], [-4.1, 5.5]], dtype=np.float64)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Three 2D arrays of shape (3, 2), int32
    a = np.ones((3, 2), dtype=np.int32)
    b = np.zeros((3, 2), dtype=np.int32)
    c = np.full((3, 2), -7, dtype=np.int32)
    input_dict = {
        "tup": (a, b, c),
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays, shape (2, 3, 4), float32
    a = np.ones((2, 3, 4), dtype=np.float32)
    b = np.full((2, 3, 4), 2.5, dtype=np.float32)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays, shape (3, 2, 2), int64
    a = np.arange(12, dtype=np.int64).reshape((3, 2, 2))
    b = np.arange(-12, 0, dtype=np.int64).reshape((3, 2, 2))
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays, shape (2, 2, 3, 2), float64
    a = np.ones((2, 2, 3, 2), dtype=np.float64)
    b = np.zeros((2, 2, 3, 2), dtype=np.float64)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Boolean 1D arrays, shape (4,)
    a = np.array([True, False, True, False], dtype=np.bool_)
    b = np.array([False, False, True, True], dtype=np.bool_)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("bool"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D arrays, shape (4, 1), float32 to float64
    a = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    b = np.array([[-1.0], [-2.0], [-3.0], [-4.0]], dtype=np.float32)
    input_dict = {
        "tup": (a, b),
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.hstack_2"] = hstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hstack_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hstack_2'.")


check_valid('jax.numpy.hstack', generated_inputs['jax.numpy.hstack_2'], lib="jax", suffix=2)
