
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squeeze_inputs():
    list_of_inputs = []

    # Input 1: 1D array with single element
    a = np.array([42.0], dtype=np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 2D array, squeeze leading dimension
    a = np.random.randn(1, 10).astype(np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 2D array, squeeze trailing dimension
    a = np.random.randn(8, 1).astype(np.int32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 3D array, squeeze multiple axes
    a = np.ones((1, 5, 1), dtype=np.float64)
    axis = [0, 2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 3D array, squeeze all dimensions
    a = np.zeros((1, 1, 1), dtype=np.bool_)
    axis = [0, 1, 2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 4D array with negative axis indexing
    a = np.random.randn(2, 1, 4, 1).astype(np.float32)
    axis = [-1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 4D array, multiple negative axes
    a = np.random.randn(3, 1, 4, 1).astype(np.int64)
    axis = [-3, -1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 5D array, complex numbers
    a = (np.random.randn(1, 2, 1, 3, 1) + 1j * np.random.randn(1, 2, 1, 3, 1)).astype(np.complex64)
    axis = [0, 2, 4]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 4D array, mixed positive and negative axis indexing
    a = np.arange(6, dtype=np.int16).reshape(1, 2, 3, 1)
    axis = [0, -1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 3D array, single negative axis
    a = np.random.randn(4, 1, 5).astype(np.float32)
    axis = [-2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 5D array, squeeze subset of unit dimensions
    a = np.random.randn(1, 1, 2, 1, 1).astype(np.float64)
    axis = [1, 3]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.squeeze_3"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.squeeze_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.squeeze_3'.")


check_valid('jax.numpy.squeeze', generated_inputs['jax.numpy.squeeze_3'], lib="jax", suffix=3)
