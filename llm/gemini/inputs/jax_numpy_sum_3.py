
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sum_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    axis = [0]
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([True, False, True, True, False])
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

    # Input 2: 2D float32 array, reduce along axis 1
    a = np.array([[1.5, -2.5, 3.0], [4.0, 5.5, -6.0]], dtype=np.float32)
    axis = [1]
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((2, 3), dtype=bool)
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

    # Input 3: 2D int32 array, reduce along axis 0
    a = np.array([[-10, 20], [30, -40], [50, 60]], dtype=np.int32)
    axis = [0]
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(10, dtype=np.int32)
    where = np.array([[True, False], [True, True], [False, True]])
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

    # Input 4: 3D float32 array, reduce over axes [0, 2]
    a = (np.arange(24, dtype=np.float32).reshape((2, 3, 4))) - 10.0
    axis = [0, 2]
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = a > 0
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

    # Input 5: 3D int32 array, reduce along axis 1
    a = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[-1, -2], [-3, -4]]],
        dtype=np.int32,
    )
    axis = [1]
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.ones((3, 2, 2), dtype=bool)
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

    # Input 6: 4D float32 array, reduce over axes [1, 3]
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = [1, 3]
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(-0.5, dtype=np.float32)
    where = a > -0.5
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

    # Input 7: 2D float32 array, reduce over all axes [0, 1]
    a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    axis = [0, 1]
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([[True, False, True], [False, True, False]])
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

    # Input 8: 1D float32 array, keepdims=True
    a = np.array([-3.14, 2.71, 0.0, -1.41], dtype=np.float32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(5.0, dtype=np.float32)
    where = np.array([True, True, False, True])
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

    # Input 9: 3D int32 array, broadcasted where condition
    a = np.ones((2, 3, 4), dtype=np.int32)
    axis = [2]
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(2, dtype=np.int32)
    where = np.array([[[True], [False], [True]]])
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

    # Input 10: 2D int32 array, reduce along axis 1
    a = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    axis = [1]
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(5, dtype=np.int32)
    where = np.array([[True, True, False], [False, True, True]])
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

    # Input 11: 4D float32 array, reduce along axis 0
    a = np.zeros((3, 2, 2, 2), dtype=np.float32)
    axis = [0]
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.zeros((3, 2, 2, 2), dtype=bool)
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


generated_inputs["jax.numpy.sum_3"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_3'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_3'], lib="jax", suffix=3)
