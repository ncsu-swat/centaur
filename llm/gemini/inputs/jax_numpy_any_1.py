
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def any_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, axis=0, keepdims=False
    a = np.array([True, False, True, False], dtype=bool)
    axis = 0
    keepdims = False
    where = np.array([True, True, False, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 2: 2D boolean array, axis=0, keepdims=True
    a = np.array(
        [[True, True, False], [False, False, False], [True, False, True]],
        dtype=bool,
    )
    axis = 0
    keepdims = True
    where = np.ones((3, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 3: 2D integer array, axis=1, keepdims=False, negative values
    a = np.array([[0, -1, 2], [0, 0, 0], [-3, 0, 5]], dtype=np.int32)
    axis = 1
    keepdims = False
    where = np.array([[True, False, True], [True, True, True], [False, True, True]])
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 4: 2D float array, negative axis=-1, keepdims=True
    a = np.array([[0.0, 1.5, 0.0], [0.0, 0.0, -2.3]], dtype=np.float32)
    axis = -1
    keepdims = True
    where = np.array([[True], [False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 5: 3D float array, axis=1, keepdims=False, broadcasting where
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 1
    keepdims = False
    where = np.ones((1, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 6: 3D boolean array, axis=-2, keepdims=True
    a = np.zeros((3, 2, 5), dtype=bool)
    a[1, 0, 2] = True
    axis = -2
    keepdims = True
    where = np.ones((3, 2, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 7: 4D integer array, axis=2, keepdims=False
    a = np.random.randint(-5, 5, size=(2, 2, 3, 2)).astype(np.int64)
    axis = 2
    keepdims = False
    where = (a > 0).astype(bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 8: 1D int array with all zeros, axis=0, keepdims=True
    a = np.zeros(10, dtype=np.int32)
    axis = 0
    keepdims = True
    where = np.ones(10, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 9: 3D array with scalar-like 1D broadcasted where, axis=0, keepdims=False
    a = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=np.uint8)
    axis = 0
    keepdims = False
    where = np.array([True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 10: 2D array with all False where mask, axis=-2, keepdims=False
    a = np.ones((4, 4), dtype=bool)
    axis = -2
    keepdims = False
    where = np.zeros((4, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 11: 4D float32 array, axis=3, keepdims=True
    a = np.zeros((2, 3, 4, 5), dtype=np.float32)
    a[0, 1, 2, 3] = 1.0
    axis = 3
    keepdims = True
    where = np.ones((2, 3, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.numpy.any_1"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_1'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_1'], lib="jax", suffix=1)
