
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive shift, axis=0
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    shift = np.array(2, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 2: 1D array, negative shift, axis=-1
    a = np.array([10, 20, 30, 40], dtype=np.int64)
    shift = np.array(-1, dtype=np.int64)
    axis = -1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 3: 2D array, float64, shift along axis 0
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    shift = np.array(1, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 4: 2D array, shift along axis 1
    a = np.random.randn(4, 5).astype(np.float32)
    shift = np.array(-3, dtype=np.int32)
    axis = 1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 5: 2D boolean array, zero shift
    a = np.array([[True, False], [False, True]], dtype=bool)
    shift = np.array(0, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 6: 3D array, shift larger than dimension size
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    shift = np.array(7, dtype=np.int64)
    axis = 2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 7: 3D complex array, negative axis index
    a = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    shift = np.array(-2, dtype=np.int32)
    axis = -2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 8: 4D uint8 array
    a = np.random.randint(0, 255, size=(2, 2, 3, 2), dtype=np.uint8)
    shift = np.array(1, dtype=np.int16)
    axis = 3
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 9: 2D array with 1D shift tensor
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    shift = np.array([2], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 10: 1D array with large negative shift
    a = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    shift = np.array([-15], dtype=np.int64)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    return list_of_inputs


generated_inputs["jax.numpy.roll_4"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_4'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_4'], lib="jax", suffix=4)
