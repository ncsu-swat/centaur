
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flip_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 1-tuple axis
    m = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 2: 2D array, flip along axis 0
    m = np.arange(6, dtype=np.float32).reshape(2, 3)
    axis = (0,)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 3: 2D array, flip along axis 1
    m = np.random.randn(4, 5).astype(np.float64)
    axis = (1,)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 4: 2D array, flip along both axes
    m = np.array([[True, False], [False, True]], dtype=np.bool_)
    axis = (0, 1)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 5: 3D array, flip along negative axis
    m = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 6: 3D array, flip along multiple axes including negative indices
    m = np.random.randn(2, 4, 3).astype(np.float32)
    axis = (0, -2)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 7: 4D array, flip along axes (1, 3)
    m = np.ones((2, 2, 3, 4), dtype=np.complex64)
    axis = (1, 3)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 8: 1D array, negative axis index
    m = np.array([-10.5, 0.0, 15.2, -3.7], dtype=np.float32)
    axis = (-1,)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 9: 5D array, flip along all axes
    m = np.arange(32, dtype=np.int16).reshape(2, 2, 2, 2, 2)
    axis = (0, 1, 2, 3, 4)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 10: 3D array with empty axis tuple
    m = np.random.randint(-100, 100, size=(3, 3, 3), dtype=np.int32)
    axis = ()
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    # Input 11: 2D array with uint8 dtype and reversed tuple order
    m = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)
    axis = (1, 0)
    list_of_inputs.append(copy.deepcopy({"m": m, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.flip_3"] = flip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flip_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flip_3'.")


check_valid('jax.numpy.flip', generated_inputs['jax.numpy.flip_3'], lib="jax", suffix=3)
