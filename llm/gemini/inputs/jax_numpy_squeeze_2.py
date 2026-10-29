
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squeeze_inputs():
    list_of_inputs = []

    # Input 1: 3D array with single axis squeeze (axis 0)
    a = np.zeros((1, 2, 3), dtype=np.float32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 3D array with multiple axes squeezed
    a = np.ones((1, 4, 1), dtype=np.int32)
    axis = (0, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: Negative axis index
    a = np.arange(5, dtype=np.float64).reshape((5, 1))
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 1D array of shape (1,) squeezed to scalar (shape ())
    a = np.array([42], dtype=np.int64)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: Empty tuple for axis (no-op squeeze)
    a = np.ones((1, 3, 1), dtype=np.float32)
    axis = ()
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 4D array with multiple negative and positive indices
    a = np.random.randn(1, 2, 1, 3).astype(np.float32)
    axis = (0, -2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: Boolean array with trailing unit dimensions
    a = np.array([[[True]], [[False]]], dtype=np.bool_)
    axis = (1, 2)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: Complex array with multiple 1-sized dimensions
    a = (np.ones((1, 1, 1, 1)) + 1j * np.ones((1, 1, 1, 1))).astype(
        np.complex64
    )
    axis = (0, 1, 2, 3)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 5D array squeezing specific middle axes
    a = np.zeros((2, 1, 3, 1, 4), dtype=np.uint8)
    axis = (1, 3)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: Array with negative values, float16
    a = np.array([-1.5, -2.5], dtype=np.float16).reshape((1, 2, 1))
    axis = (-3, -1)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: Array with 0-size dimension alongside 1-size dimension
    a = np.empty((0, 1, 5), dtype=np.float32)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.squeeze_2"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.squeeze_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.squeeze_2'.")


check_valid('jax.numpy.squeeze', generated_inputs['jax.numpy.squeeze_2'], lib="jax", suffix=2)
