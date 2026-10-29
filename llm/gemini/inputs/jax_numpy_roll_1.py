
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive shift, axis 0
    a = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    shift = 2
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 2: 1D array, negative shift, axis 0
    a = np.array([10.5, 20.5, 30.5, 40.5], dtype=np.float32)
    shift = -1
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 3: 2D array, positive shift along axis 0
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    shift = 1
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 4: 2D array, positive shift along axis 1
    a = np.random.randn(4, 5).astype(np.float64)
    shift = 3
    axis = 1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 5: 2D array, negative axis index
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    shift = -2
    axis = -1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 6: 3D array, shift larger than axis length
    a = np.ones((2, 3, 4), dtype=np.int16)
    shift = 7
    axis = 1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 7: 3D array, zero shift
    a = np.random.randn(3, 4, 2).astype(np.float32)
    shift = 0
    axis = 2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 8: 3D array with negative axis and negative shift
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    shift = -5
    axis = -2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 9: 4D boolean array
    a = np.array([True, False] * 12, dtype=bool).reshape(2, 2, 2, 3)
    shift = 1
    axis = 3
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 10: Complex array, 2D
    a = (np.arange(8) + 1j * np.arange(8)).reshape(4, 2).astype(np.complex64)
    shift = -2
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    return list_of_inputs


generated_inputs["jax.numpy.roll_1"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_1'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_1'], lib="jax", suffix=1)
