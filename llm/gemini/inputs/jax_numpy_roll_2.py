
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roll_inputs():
    list_of_inputs = []

    # Case 1: 1D float32 array, simple shift
    a = np.arange(10, dtype=np.float32)
    shift = [2]
    axis = [0]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 2: 2D float32 array, shifts on both axes
    a = np.random.randn(4, 5).astype(np.float32)
    shift = [1, -2]
    axis = [0, 1]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 3: 2D int32 array, shift only on axis 1
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    shift = [2]
    axis = [1]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 4: 3D float64 array, shift on all three axes
    a = np.random.randn(2, 3, 4).astype(np.float64)
    shift = [1, -1, 2]
    axis = [0, 1, 2]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 5: 3D int32 array, shifting subset of axes
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    shift = [-1, 2]
    axis = [0, 2]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 6: 4D float32 array, shifting 3 out of 4 axes
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    shift = [1, -1, 1]
    axis = [0, 2, 3]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 7: 2D bool array, shifting on axis 0
    a = (np.random.randn(5, 5) > 0).astype(bool)
    shift = [-3]
    axis = [0]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 8: 1D int64 array, large shift
    a = np.arange(100, dtype=np.int64)
    shift = [45]
    axis = [0]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 9: 3D float32 array, zero shifts (no-op)
    a = np.random.randn(3, 3, 3).astype(np.float32)
    shift = [0, 0]
    axis = [1, 2]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

    # Case 10: 2D complex64 array, shifts on both axes
    a = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex64)
    shift = [-1, 1]
    axis = [0, 1]
    list_of_inputs.append({"a": a, "shift": shift, "axis": axis})

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
