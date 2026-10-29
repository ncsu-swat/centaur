
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([0, 1, -2, 0, 3, 0, -5], dtype=np.int32)
    axis = (0,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 2: 2D float array, count along axis 0
    a = np.array([[0.0, 1.5, 0.0], [-2.3, 0.0, 4.1]], dtype=np.float32)
    axis = (0,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 3: 2D float array, count along axis 1
    a = np.array(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [2.5, -3.2, 0.0]],
        dtype=np.float64,
    )
    axis = (1,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 4: 2D integer array, reduce multiple axes
    a = np.array([[0, 2, 0], [4, 0, 6]], dtype=np.int64)
    axis = (0, 1)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 5: 3D boolean array, reduce over negative axis
    a = np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=bool)
    axis = (-1,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 6: 3D float array, reduce over axes (0, 2)
    a = np.zeros((3, 4, 5), dtype=np.float32)
    a[0, 1, 2] = 1.0
    a[2, 3, 4] = -7.5
    axis = (0, 2)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 7: 3D complex array, keepdims True
    a = np.array(
        [[[0 + 0j, 1 + 2j], [0 + 1j, 0 + 0j]], [[-1 + 0j, 0 + 0j], [3 - 2j, 0 + 0j]]],
        dtype=np.complex64,
    )
    axis = (1, 2)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 8: 4D integer array, single negative axis
    a = np.random.randint(-2, 3, size=(2, 3, 4, 2)).astype(np.int16)
    axis = (-2,)
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 9: 4D float array, empty tuple axis (count across zero axes)
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = ()
    keepdims = False
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 10: 2D unsigned int array with all zeros
    a = np.zeros((5, 5), dtype=np.uint8)
    axis = (0,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    # Input 11: 1D array, keepdims True
    a = np.array([0, -1, 0, 2, 0], dtype=np.int8)
    axis = (0,)
    keepdims = True
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims})
    )

    return list_of_inputs


generated_inputs["jax.numpy.count_nonzero_2"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_2'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_2'], lib="jax", suffix=2)
