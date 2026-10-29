
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def size_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0
    a = np.arange(10, dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 1D array, negative axis=-1
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 2D array, axis=0
    a = np.ones((4, 5), dtype=np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 2D array, axis=1
    a = np.zeros((3, 7), dtype=bool)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 2D array, negative axis=-2
    a = np.empty((6, 2), dtype=np.complex64)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 3D array, axis=2
    a = np.random.randn(2, 3, 4).astype(np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 3D array, negative axis=-3
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    axis = -3
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 4D array, axis=1
    a = np.ones((2, 4, 3, 5), dtype=np.uint8)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 4D array, negative axis=-4
    a = np.zeros((5, 1, 2, 3), dtype=np.float32)
    axis = -4
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 5D array, axis=4
    a = np.ones((1, 2, 3, 4, 5), dtype=np.int16)
    axis = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 2D empty dimension array, axis=0
    a = np.empty((0, 8), dtype=np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 12: 2D empty dimension array, axis=1
    a = np.empty((0, 8), dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.size_1"] = size_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.size_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.size_1'.")


check_valid('jax.numpy.size', generated_inputs['jax.numpy.size_1'], lib="jax", suffix=1)
