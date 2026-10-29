
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis expansion at leading position
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 1D array, single axis expansion at trailing position
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 1D array, multiple axes expansion
    a = np.array([True, False, True], dtype=np.bool_)
    axis = (0, 1, 3)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 2D array, expanding between existing axes
    a = np.random.randn(3, 4).astype(np.float64)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 2D array, negative axis index
    a = np.arange(6, dtype=np.int64).reshape((2, 3))
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 0D scalar array, expand to 2D
    a = np.array(42, dtype=np.int16)
    axis = (0, 1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 3D array, negative axes
    a = np.ones((2, 3, 4), dtype=np.uint8)
    axis = (-2, -1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 4D array, empty axis tuple
    a = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    axis = ()
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 2D array, expanding multiple axes
    a = np.random.uniform(-1, 1, size=(5, 5)).astype(np.float32)
    axis = (0, 2, 4)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 1D empty array
    a = np.array([], dtype=np.float32)
    axis = (0, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 3D array with complex numbers, leading and trailing axes
    a = (np.arange(8) + 1j * np.arange(8)).reshape((2, 2, 2)).astype(np.complex128)
    axis = (0, 4)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.expand_dims_2"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_2'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_2'], lib="jax", suffix=2)
