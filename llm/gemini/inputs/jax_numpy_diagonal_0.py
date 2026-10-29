
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def diagonal_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D square matrix, main diagonal
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    list_of_inputs.append({"a": a, "offset": 0, "axis1": 0, "axis2": 1})

    # Input 2: 2D matrix with positive offset (super-diagonal)
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    list_of_inputs.append({"a": a, "offset": 1, "axis1": 0, "axis2": 1})

    # Input 3: 2D matrix with negative offset (sub-diagonal)
    a = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"a": a, "offset": -2, "axis1": 0, "axis2": 1})

    # Input 4: 2D rectangular matrix with swapped axes
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    list_of_inputs.append({"a": a, "offset": 0, "axis1": 1, "axis2": 0})

    # Input 5: 3D tensor, diagonal along axes 0 and 1
    a = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    list_of_inputs.append({"a": a, "offset": 0, "axis1": 0, "axis2": 1})

    # Input 6: 3D tensor, diagonal along axes 1 and 2 with offset
    a = np.random.randn(2, 4, 5).astype(np.float32)
    list_of_inputs.append({"a": a, "offset": -1, "axis1": 1, "axis2": 2})

    # Input 7: 3D tensor using negative axis indices
    a = np.arange(30, dtype=np.int16).reshape(2, 5, 3)
    list_of_inputs.append({"a": a, "offset": 1, "axis1": -2, "axis2": -1})

    # Input 8: 4D tensor with float16 dtype
    a = np.random.randn(2, 3, 4, 5).astype(np.float16)
    list_of_inputs.append({"a": a, "offset": 0, "axis1": 0, "axis2": 2})

    # Input 9: 4D tensor with complex numbers
    a = (np.random.randn(2, 4, 3, 4) + 1j * np.random.randn(2, 4, 3, 4)).astype(
        np.complex64
    )
    list_of_inputs.append({"a": a, "offset": -1, "axis1": 1, "axis2": 3})

    # Input 10: 5D tensor with boolean dtype
    a = np.random.rand(2, 2, 3, 3, 2) > 0.5
    list_of_inputs.append({"a": a, "offset": 0, "axis1": 2, "axis2": 3})

    # Input 11: 2D boolean array with large positive offset
    a = np.ones((6, 6), dtype=bool)
    list_of_inputs.append({"a": a, "offset": 3, "axis1": 0, "axis2": 1})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.diagonal"] = diagonal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.diagonal'.")


check_valid('jax.numpy.diagonal', generated_inputs['jax.numpy.diagonal'], lib="jax", suffix=0)
