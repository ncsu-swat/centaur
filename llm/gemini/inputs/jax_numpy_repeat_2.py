
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def repeat_inputs():
    list_of_inputs = []

    # Input 1: 1D array, repeat=2, axis=0
    a = np.array([1, 2, 3], dtype=np.int32)
    repeats = 2
    axis = 0
    total_repeat_length = 6
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 2: 2D array, repeat=3, axis=1
    a = np.random.randn(2, 4).astype(np.float32)
    repeats = 3
    axis = 1
    total_repeat_length = 12
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 3: 2D array, negative axis=-1
    a = np.random.randn(3, 5).astype(np.float64)
    repeats = 2
    axis = -1
    total_repeat_length = 10
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 4: 3D array, axis=0
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    repeats = 4
    axis = 0
    total_repeat_length = 8
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 5: Truncated output (total_repeat_length < repeats * dim_size)
    a = np.array([[10, 20], [30, 40]], dtype=np.int16)
    repeats = 3
    axis = 0
    total_repeat_length = 5
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 6: Padded/extended output (total_repeat_length > repeats * dim_size)
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    repeats = 2
    axis = 1
    total_repeat_length = 6
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 7: repeats = 0, total_repeat_length = 0
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    repeats = 0
    axis = 0
    total_repeat_length = 0
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 8: 4D array, axis=2
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    repeats = 2
    axis = 2
    total_repeat_length = 8
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 9: Boolean tensor, negative axis=-2
    a = np.array([[[True, False], [False, True]]], dtype=bool)
    repeats = 3
    axis = -2
    total_repeat_length = 6
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    # Input 10: Complex128 tensor
    a = (np.random.randn(3, 2) + 1j * np.random.randn(3, 2)).astype(
        np.complex128
    )
    repeats = 1
    axis = 0
    total_repeat_length = 3
    list_of_inputs.append(
        {
            "a": a,
            "repeats": repeats,
            "axis": axis,
            "total_repeat_length": total_repeat_length,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.repeat_2"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_2'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_2'], lib="jax", suffix=2)
