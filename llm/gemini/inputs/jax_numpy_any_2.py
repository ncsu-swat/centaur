
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def any_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, reduce over axis 0
    a = np.array([True, False, False, True], dtype=bool)
    axis = (0,)
    keepdims = False
    where = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 2: 2D float32 array, reduce along axis 1 with keepdims=True
    a = np.array([[-1.5, 0.0, 2.3], [0.0, 0.0, -0.1]], dtype=np.float32)
    axis = (1,)
    keepdims = True
    where = np.array([[True, False, True], [False, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 3: 2D int32 array, reduce over both axes (0, 1)
    a = np.array([[0, 2], [-3, 0]], dtype=np.int32)
    axis = (0, 1)
    keepdims = False
    where = np.array([[True, True], [True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 4: 3D float64 array, reduce along non-adjacent axes (0, 2), broadcasted where
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = (0, 2)
    keepdims = True
    where = np.ones((1, 3, 1), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 5: 3D int16 array with negative axis, keepdims=False
    a = np.zeros((3, 4, 5), dtype=np.int16)
    a[1, 2, 3] = 5
    axis = (-1,)
    keepdims = False
    where = np.ones((3, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 6: 4D boolean array, reduce over middle axes (1, 2)
    a = np.random.choice([True, False], size=(2, 3, 3, 2))
    axis = (1, 2)
    keepdims = True
    where = np.random.choice([True, False], size=(2, 3, 3, 2))
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 7: 1D complex64 array with keepdims=True, negative axis
    a = np.array([0 + 0j, 1 - 2j, 0 + 0j, -3 + 0j], dtype=np.complex64)
    axis = (-1,)
    keepdims = True
    where = np.array([True, True, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 8: 2D uint8 array, reduce along axis 0, broadcasted 1D where
    a = np.array([[0, 1, 0], [2, 0, 4]], dtype=np.uint8)
    axis = (0,)
    keepdims = False
    where = np.array([True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 9: 3D boolean array, reduce all axes (0, 1, 2), scalar-like broadcast where
    a = np.zeros((2, 2, 2), dtype=bool)
    axis = (0, 1, 2)
    keepdims = False
    where = np.array([[[True]]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 10: 4D int8 array with multiple negative axes, keepdims=True
    a = np.random.randint(-10, 10, size=(2, 3, 4, 2), dtype=np.int8)
    axis = (-3, -1)
    keepdims = True
    where = np.ones((2, 1, 4, 1), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.numpy.any_2"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_2'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_2'], lib="jax", suffix=2)
