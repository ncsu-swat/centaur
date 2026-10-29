
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def size_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single element tuple axis
    a = np.arange(10, dtype=np.int32)
    axis = (0,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 2: 1D array, negative axis in tuple
    a = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    axis = (-1,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 3: 2D array, axis (0,)
    a = np.ones((4, 5), dtype=np.float64)
    axis = (0,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 4: 2D array, axis (1,)
    a = np.zeros((3, 7), dtype=np.int64)
    axis = (1,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 5: 2D array, both axes in tuple
    a = np.random.randn(6, 8).astype(np.float32)
    axis = (0, 1)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 6: 2D array, negative axes
    a = np.ones((5, 10), dtype=np.bool_)
    axis = (-2, -1)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 7: 3D array, axis (1, 2)
    a = np.random.randint(-10, 10, size=(2, 4, 6), dtype=np.int16)
    axis = (1, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 8: 3D array, non-consecutive axes
    a = np.random.randn(3, 4, 5).astype(np.complex64)
    axis = (0, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 9: 4D array, all axes
    a = np.ones((2, 3, 4, 5), dtype=np.uint8)
    axis = (0, 1, 2, 3)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 10: 4D array, subset of axes
    a = np.random.randn(2, 5, 3, 4).astype(np.float32)
    axis = (1, 3)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 11: 5D array, single negative axis
    a = np.zeros((1, 2, 3, 4, 5), dtype=np.int8)
    axis = (-3,)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    # Input 12: 2D array with a zero-length dimension
    a = np.empty((0, 8), dtype=np.float32)
    axis = (0, 1)
    list_of_inputs.append({"a": copy.deepcopy(a), "axis": copy.deepcopy(axis)})

    return list_of_inputs


generated_inputs["jax.numpy.size_2"] = size_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.size_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.size_2'.")


check_valid('jax.numpy.size', generated_inputs['jax.numpy.size_2'], lib="jax", suffix=2)
