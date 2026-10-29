
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def repeat_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, exact total_repeat_length
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    repeats = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 0,
            "total_repeat_length": 6,
        }
    )

    # Input 2: 2D int32 array, axis 0
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    repeats = np.array([3, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 0,
            "total_repeat_length": 5,
        }
    )

    # Input 3: 2D float64 array, axis 1
    a = np.array([[1.5, -2.5], [3.0, 4.0]], dtype=np.float64)
    repeats = np.array([2, 4], dtype=np.int64)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 1,
            "total_repeat_length": 6,
        }
    )

    # Input 4: Truncated output (total_repeat_length < sum(repeats)), negative axis
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    repeats = np.array([2, 2, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": -1,
            "total_repeat_length": 4,
        }
    )

    # Input 5: Extended output (total_repeat_length > sum(repeats))
    a = np.array([0.1, -0.5, 2.3, -4.1], dtype=np.float32)
    repeats = np.array([1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 0,
            "total_repeat_length": 7,
        }
    )

    # Input 6: 3D float32 array, axis 1
    a = np.ones((2, 3, 4), dtype=np.float32)
    repeats = np.array([1, 2, 1], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 1,
            "total_repeat_length": 4,
        }
    )

    # Input 7: 3D int16 array with negative axis
    a = np.arange(24, dtype=np.int16).reshape(3, 2, 4)
    repeats = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": -3,
            "total_repeat_length": 6,
        }
    )

    # Input 8: Boolean array
    a = np.array([[True, False], [False, True]], dtype=bool)
    repeats = np.array([3, 1], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 0,
            "total_repeat_length": 4,
        }
    )

    # Input 9: Complex64 array
    a = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
    repeats = np.array([1, 3, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 0,
            "total_repeat_length": 6,
        }
    )

    # Input 10: 4D float32 array, axis 2
    a = np.zeros((2, 2, 3, 2), dtype=np.float32)
    repeats = np.array([2, 1, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": 2,
            "total_repeat_length": 5,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.repeat_1"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_1'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_1'], lib="jax", suffix=1)
