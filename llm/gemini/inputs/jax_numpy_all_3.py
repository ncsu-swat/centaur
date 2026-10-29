
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def all_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce axis 0
    a = np.array([True, True, False, True], dtype=bool)
    axis = [0]
    keepdims = False
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 2: 2D boolean array, keepdims=True
    a = np.array([[True, True], [False, True]], dtype=bool)
    axis = [0]
    keepdims = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 3: 2D integer array, axis=[1]
    a = np.array([[1, 2, 3], [0, 5, 6]], dtype=np.int32)
    axis = [1]
    keepdims = False
    where = np.array([[True, True, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 4: 2D float array, negative axis [-1]
    a = np.array([[1.0, -2.5], [0.0, 4.2]], dtype=np.float32)
    axis = [-1]
    keepdims = True
    where = np.array([[True, True], [True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 5: 3D float array, multiple axes [0, 2]
    a = np.ones((2, 3, 4), dtype=np.float64)
    axis = [0, 2]
    keepdims = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 6: 3D integer array with broadcasting where
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    axis = [1]
    keepdims = True
    where = np.ones((1, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 7: 4D boolean array, multiple axes [1, 3]
    a = np.random.choice([True, False], size=(2, 2, 2, 2))
    axis = [1, 3]
    keepdims = False
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 8: 2D array, reduce all axes [0, 1]
    a = np.array([[10, 20], [30, 40]], dtype=np.int16)
    axis = [0, 1]
    keepdims = True
    where = np.array([[True, False], [True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 9: 1D float array with negative numbers, axis [0]
    a = np.array([-1.0, -0.5, 0.0, 1.5], dtype=np.float32)
    axis = [0]
    keepdims = False
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 10: 3D complex array, axis [2]
    a = np.array(
        [[[1 + 1j, 2 + 0j], [0 + 0j, 3 - 1j]], [[-1 + 2j, 4 + 0j], [5 + 1j, 6 + 0j]]],
        dtype=np.complex64,
    )
    axis = [2]
    keepdims = False
    where = np.ones((2, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 11: 3D boolean array, negative axes [-3, -1]
    a = np.ones((3, 2, 4), dtype=bool)
    axis = [-3, -1]
    keepdims = True
    where = np.ones((3, 1, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.numpy.all_3"] = all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.all_3'.")


check_valid('jax.numpy.all', generated_inputs['jax.numpy.all_3'], lib="jax", suffix=3)
