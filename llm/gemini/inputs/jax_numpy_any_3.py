
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def any_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    a = np.array([False, True, False, False], dtype=bool)
    axis = [0]
    keepdims = False
    where = np.array([True, True, False, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 2: 2D float32 array with negative and positive values
    a = np.array([[-1.5, 0.0, 2.3], [0.0, 0.0, -0.1]], dtype=np.float32)
    axis = [1]
    keepdims = True
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 3: 2D integer array, reduce along multiple axes
    a = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int32)
    axis = [0, 1]
    keepdims = False
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 4: 3D float array with broadcastable where
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [0, 2]
    keepdims = True
    where = np.random.choice([True, False], size=(1, 3, 1))
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 5: 1D integer array with negative index axis
    a = np.array([-5, 0, 10, -2], dtype=np.int64)
    axis = [-1]
    keepdims = True
    where = np.array([False, True, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 6: 4D boolean array, reduce over non-consecutive axes
    a = np.random.choice([True, False], size=(2, 3, 2, 4))
    axis = [1, 3]
    keepdims = False
    where = np.random.choice([True, False], size=(2, 1, 2, 1))
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 7: 2D int8 array
    a = np.array([[0, -1], [1, 0]], dtype=np.int8)
    axis = [0]
    keepdims = False
    where = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 8: 3D array with negative axis
    a = np.zeros((3, 4, 5), dtype=np.float32)
    a[1, 2, 3] = 1.0
    axis = [-2]
    keepdims = True
    where = np.ones((3, 4, 5), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 9: 2D float16 array with zeros
    a = np.array([[0.0, 0.0], [0.0, 3.14]], dtype=np.float16)
    axis = [1]
    keepdims = False
    where = np.array([[True, True], [True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 10: 3D uint8 array
    a = np.array([[[1, 0], [0, 0]], [[0, 0], [2, 0]]], dtype=np.uint8)
    axis = [0, 1]
    keepdims = True
    where = np.array([[[True, True], [True, False]], [[False, True], [True, True]]])
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    # Input 11: 2D array, broadcasting 1D where
    a = np.array([[10, 20, 30], [0, -1, 0]], dtype=np.int32)
    axis = [-1]
    keepdims = False
    where = np.array([[True, False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": axis, "keepdims": keepdims, "where": where})
    )

    return list_of_inputs


generated_inputs["jax.numpy.any_3"] = any_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.any_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.any_3'.")


check_valid('jax.numpy.any', generated_inputs['jax.numpy.any_3'], lib="jax", suffix=3)
