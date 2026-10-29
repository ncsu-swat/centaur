
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive shift
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shift = [2]
    axis = [0]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 2: 1D array, negative shift
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    shift = [-1]
    axis = [0]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 3: 2D array, shift along one axis
    a = np.arange(12, dtype=np.int64).reshape((3, 4))
    shift = [1]
    axis = [0]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 4: 2D array, shifts along both axes
    a = np.random.randn(4, 5).astype(np.float64)
    shift = [2, -1]
    axis = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 5: 2D array, negative axis index
    a = np.arange(9, dtype=np.uint8).reshape((3, 3))
    shift = [-2]
    axis = [-1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 6: 3D array, shift along multiple axes with negative shift values
    a = np.random.randn(2, 3, 4).astype(np.float32)
    shift = [-1, 2]
    axis = [0, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 7: 3D array, shifts along all three axes
    a = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    shift = [1, 0, -2]
    axis = [0, 1, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 8: 3D boolean array, zero shifts
    a = np.array([[[True, False], [False, True]]], dtype=np.bool_)
    shift = [0, 0]
    axis = [1, 2]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 9: 4D array, complex dtype
    a = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(
        np.complex64
    )
    shift = [1, -1, 2, -2]
    axis = [0, 1, 2, 3]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 10: 2D array, large shift values exceeding dimension sizes
    a = np.arange(6, dtype=np.float32).reshape((2, 3))
    shift = [10, -15]
    axis = [0, 1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    # Input 11: 4D array, negative axis indices
    a = np.zeros((3, 4, 5, 2), dtype=np.int16)
    shift = [3, -2]
    axis = [-3, -1]
    list_of_inputs.append(
        copy.deepcopy({"a": a, "shift": shift, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.roll_2"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_2'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_2'], lib="jax", suffix=2)
