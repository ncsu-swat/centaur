
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive shift
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    shift = (2,)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 2: 1D array, int32, negative shift
    a = np.arange(10, dtype=np.int32)
    shift = (-3,)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 3: 2D array, float64, shift along both axes
    a = np.arange(12, dtype=np.float64).reshape((3, 4))
    shift = (1, -2)
    axis = (0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 4: 2D array, bool, shift along one axis
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    shift = (1,)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 5: 3D array, complex64, shift along axis 0 and 2
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    shift = (-1, 2)
    axis = (0, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 6: 3D array, int64, roll along negative axes
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    shift = (2, -1)
    axis = (-3, -1)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 7: 4D array, uint8, shift along all axes
    a = np.arange(16, dtype=np.uint8).reshape((2, 2, 2, 2))
    shift = (1, 0, -1, 2)
    axis = (0, 1, 2, 3)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 8: 2D array, float32, zero shifts
    a = np.ones((5, 5), dtype=np.float32)
    shift = (0, 0)
    axis = (0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 9: 3D array, float16, large shift values exceeding dimension sizes
    a = np.random.randn(3, 4, 5).astype(np.float16)
    shift = (10, -15, 23)
    axis = (0, 1, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    # Input 10: 1D array, int16, negative axis index
    a = np.array([10, 20, 30, 40], dtype=np.int16)
    shift = (-1,)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "shift": shift, "axis": axis}))

    return list_of_inputs

generated_inputs["jax.numpy.roll_3"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_3'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_3'], lib="jax", suffix=3)
