
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def custom_pad(row, pad_width, iaxis, kwargs):
    before, after = pad_width
    return row


def pad_inputs():
    list_of_inputs = []

    # When mode is a custom callable, JAX passes all arbitrary kwargs into kwargs dict
    # without checking for unsupported keyword arguments.
    callable_mode = custom_pad

    # Input 1: 1D float32 array
    array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 2),),
            "mode": callable_mode,
            "constant_values": 0.0,
            "stat_length": ((1, 1),),
            "end_values": 0.0,
            "reflect_type": "even",
        }
    )

    # Input 2: 1D float64 array, negative values
    array = np.array([-2.5, 0.0, 2.5, 5.0], dtype=np.float64)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((2, 2),),
            "mode": callable_mode,
            "constant_values": -1.0,
            "stat_length": ((1, 1),),
            "end_values": -1.0,
            "reflect_type": "odd",
        }
    )

    # Input 3: 2D array, asymmetric padding
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 2), (2, 1)),
            "mode": callable_mode,
            "constant_values": 99.0,
            "stat_length": ((1, 1), (1, 1)),
            "end_values": 10.0,
            "reflect_type": "even",
        }
    )

    # Input 4: 2D array, float64
    array = np.ones((4, 4), dtype=np.float64)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 1), (1, 1)),
            "mode": callable_mode,
            "constant_values": 0.5,
            "stat_length": ((1, 1), (1, 1)),
            "end_values": -5.5,
            "reflect_type": "even",
        }
    )

    # Input 5: 3D array
    array = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 1), (1, 1), (2, 2)),
            "mode": callable_mode,
            "constant_values": 7.5,
            "stat_length": ((1, 1), (1, 1), (1, 1)),
            "end_values": 2.0,
            "reflect_type": "even",
        }
    )

    # Input 6: 1D array, large padding
    array = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((3, 3),),
            "mode": callable_mode,
            "constant_values": 0.0,
            "stat_length": ((1, 1),),
            "end_values": 5.0,
            "reflect_type": "odd",
        }
    )

    # Input 7: 2D array, small dimensions
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 1), (1, 1)),
            "mode": callable_mode,
            "constant_values": 3.0,
            "stat_length": ((1, 1), (1, 1)),
            "end_values": 0.0,
            "reflect_type": "even",
        }
    )

    # Input 8: 1D single-element array
    array = np.array([42.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 1),),
            "mode": callable_mode,
            "constant_values": 1.0,
            "stat_length": ((1, 1),),
            "end_values": 2.0,
            "reflect_type": "even",
        }
    )

    # Input 9: 2D array, zero padding on one side
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((0, 2), (1, 0)),
            "mode": callable_mode,
            "constant_values": -3.0,
            "stat_length": ((1, 1), (1, 1)),
            "end_values": 4.0,
            "reflect_type": "odd",
        }
    )

    # Input 10: 3D array with various shapes
    array = np.zeros((3, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        {
            "array": array,
            "pad_width": ((1, 0), (0, 1), (1, 1)),
            "mode": callable_mode,
            "constant_values": 8.0,
            "stat_length": ((1, 1), (1, 1), (1, 1)),
            "end_values": -1.5,
            "reflect_type": "even",
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.pad_2"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.pad_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.pad_2'.")


check_valid('jax.numpy.pad', generated_inputs['jax.numpy.pad_2'], lib="jax", suffix=2)
