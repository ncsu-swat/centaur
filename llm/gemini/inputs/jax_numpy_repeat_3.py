
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def repeat_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, exact match total_repeat_length
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    repeats = np.array([2, 1, 3], dtype=np.int32)
    axis = 0
    total_repeat_length = 6
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 2: 2D int32 array, axis 1, exact match
    a = np.arange(8, dtype=np.int32).reshape(2, 4)
    repeats = np.array([1, 2, 1, 2], dtype=np.int32)
    axis = 1
    total_repeat_length = 6
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 3: 2D float64 array, negative axis -1, truncated total_repeat_length
    a = np.array([[1.5, -2.5, 3.5], [-4.5, 5.5, -6.5]], dtype=np.float64)
    repeats = np.array([3, 2, 1], dtype=np.int32)
    axis = -1
    total_repeat_length = 4
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 4: 3D int64 array, axis 0, padded total_repeat_length
    a = np.arange(12, dtype=np.int64).reshape(2, 2, 3)
    repeats = np.array([2, 3], dtype=np.int32)
    axis = 0
    total_repeat_length = 7
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 5: 3D bool array, axis 2
    a = np.array([[[True, False], [False, True]]], dtype=bool)
    repeats = np.array([2, 1], dtype=np.int32)
    axis = 2
    total_repeat_length = 3
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 6: 1D int16 array with zero repeats, negative axis -1
    a = np.array([-5, 10, -15, 20], dtype=np.int16)
    repeats = np.array([0, 2, 0, 1], dtype=np.int32)
    axis = -1
    total_repeat_length = 3
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 7: 4D float32 array, axis 1
    a = np.ones((2, 3, 2, 2), dtype=np.float32)
    repeats = np.array([1, 1, 2], dtype=np.int32)
    axis = 1
    total_repeat_length = 4
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 8: 2D complex64 array, axis 0
    a = np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex64)
    repeats = np.array([2, 2], dtype=np.int32)
    axis = 0
    total_repeat_length = 4
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 9: 1D float32 single-element array, axis 0
    a = np.array([3.14159], dtype=np.float32)
    repeats = np.array([5], dtype=np.int32)
    axis = 0
    total_repeat_length = 5
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 10: 3D uint8 array, negative axis -2, padded total_repeat_length
    a = np.zeros((2, 3, 2), dtype=np.uint8)
    repeats = np.array([1, 2, 1], dtype=np.int32)
    axis = -2
    total_repeat_length = 6
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    # Input 11: 2D float16 array, negative axis -2, truncated total_repeat_length
    a = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float16)
    repeats = np.array([2, 1, 3, 1], dtype=np.int32)
    axis = -2
    total_repeat_length = 5
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "repeats": repeats,
                "axis": axis,
                "total_repeat_length": total_repeat_length,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.repeat_3"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_3'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_3'], lib="jax", suffix=3)
