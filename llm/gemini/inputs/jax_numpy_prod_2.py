
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    a = np.array([2.0, -3.0, 4.0], dtype=np.float32)
    axis = (0,)
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, False, True], dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 2: 2D int64 array, reduce along axis 1 with keepdims=True
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    axis = (1,)
    dtype = np.dtype(np.int64)
    keepdims = True
    initial = np.array(2, dtype=np.int64)
    where = np.array([[True, True, False], [False, True, True]], dtype=bool)
    promote_integers = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 3: 2D float64 array with negative numbers, reduce all axes
    a = np.array([[-1.5, 2.0], [3.0, -0.5]], dtype=np.float64)
    axis = (0, 1)
    dtype = np.dtype(np.float64)
    keepdims = False
    initial = np.array(-1.0, dtype=np.float64)
    where = np.array([[True, True], [False, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 4: 3D int32 array, reduce along multiple axes (0, 2)
    a = np.arange(1, 13, dtype=np.int32).reshape((2, 3, 2))
    axis = (0, 2)
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.ones((2, 3, 2), dtype=bool)
    promote_integers = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 5: 1D int32 array with negative values, keepdims=True
    a = np.array([-2, -3, -4, 5], dtype=np.int32)
    axis = (0,)
    dtype = np.dtype(np.int64)
    keepdims = True
    initial = np.array(1, dtype=np.int64)
    where = np.array([True, True, True, True], dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 6: 3D float32 array, reduce along axis 1
    a = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]], dtype=np.float32
    )
    axis = (1,)
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.5, dtype=np.float32)
    where = np.array(
        [[[True, False], [True, True]], [[False, True], [True, False]]],
        dtype=bool,
    )
    promote_integers = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 7: 4D float32 array, reduce along axes (2, 3)
    a = np.ones((2, 2, 3, 3), dtype=np.float32) * 1.5
    axis = (2, 3)
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 3, 3), dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 8: 2D complex64 array, reduce along axis 0
    a = np.array(
        [[1.0 + 1.0j, 2.0 - 1.0j], [0.5 + 0.5j, 1.0 + 0.0j]],
        dtype=np.complex64,
    )
    axis = (0,)
    dtype = np.dtype(np.complex64)
    keepdims = False
    initial = np.array(1.0 + 0.0j, dtype=np.complex64)
    where = np.array([[True, False], [True, True]], dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 9: 4D int32 array, reduce along axes (1, 2)
    a = np.array([[[[2], [3]], [[4], [5]]]], dtype=np.int32)
    axis = (1, 2)
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(3, dtype=np.int32)
    where = np.ones((1, 2, 2, 1), dtype=bool)
    promote_integers = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    # Input 10: 1D float64 array with masked elements, keepdims=True
    a = np.array([0.5, 2.0, 4.0, 0.25, 8.0], dtype=np.float64)
    axis = (0,)
    dtype = np.dtype(np.float64)
    keepdims = True
    initial = np.array(2.0, dtype=np.float64)
    where = np.array([False, True, True, False, True], dtype=bool)
    promote_integers = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
                "promote_integers": promote_integers,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.prod_2"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_2'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_2'], lib="jax", suffix=2)
