
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def size_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis
    a = np.arange(10, dtype=np.int32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 2D float32 array, axis 0
    a = np.ones((4, 5), dtype=np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 2D int64 array, axis 1
    a = np.zeros((3, 7), dtype=np.int64)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 2D bool array, both axes
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis = [0, 1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 3D float64 array, negative axis index
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 3D int32 array, multiple non-adjacent axes
    a = np.random.randint(-10, 10, size=(4, 6, 8), dtype=np.int32)
    axis = [0, 2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 4D complex64 array, mixed positive and negative axes
    a = (np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)).astype(
        np.complex64
    )
    axis = [1, -1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 4D float32 array, all axes
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    axis = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 5D uint8 array, subset of axes
    a = np.random.randint(0, 255, size=(2, 3, 2, 4, 2), dtype=np.uint8)
    axis = [0, 2, 4]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 3D float16 array, negative axis
    a = np.random.randn(5, 7, 9).astype(np.float16)
    axis = [-2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 2D array with a zero dimension
    a = np.empty((0, 5), dtype=np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 12: 2D array with a zero dimension, counting non-zero axis
    a = np.empty((0, 5), dtype=np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.size_3"] = size_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.size_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.size_3'.")


check_valid('jax.numpy.size', generated_inputs['jax.numpy.size_3'], lib="jax", suffix=3)
