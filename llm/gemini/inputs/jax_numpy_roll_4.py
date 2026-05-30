
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roll_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array, positive shift
    a = np.arange(10, dtype=np.float32)
    shift = np.array(2, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 2: 2D int32 array, negative shift on axis 1
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    shift = np.array(-1, dtype=np.int32)
    axis = 1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 3: 3D float64 array, shift represented as 1D array of size 1
    a = np.random.randn(2, 3, 4).astype(np.float64)
    shift = np.array([2], dtype=np.int64)
    axis = 2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 4: 4D float32 array, negative axis index
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    shift = np.array(-2, dtype=np.int32)
    axis = -1
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 5: Boolean 1D array, shift larger than array size
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    shift = np.array(7, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 6: 2D float32 array, shift is zero
    a = np.random.randn(5, 5).astype(np.float32)
    shift = np.array(0, dtype=np.int32)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 7: 3D int16 array, negative shift on axis 0
    a = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    shift = np.array(-3, dtype=np.int16)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 8: 1D uint8 array, shift of 0D array with int64 type
    a = np.array([10, 20, 30, 40], dtype=np.uint8)
    shift = np.array(1, dtype=np.int64)
    axis = 0
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 9: 2D float64 array, shift of size 1 array, negative axis
    a = np.random.randn(4, 4).astype(np.float64)
    shift = np.array([-2], dtype=np.int32)
    axis = -2
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Input 10: 4D int32 array, larger dimensions
    a = np.arange(120, dtype=np.int32).reshape(2, 3, 4, 5)
    shift = np.array(4, dtype=np.int32)
    axis = 3
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
