
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis [0]
    a = np.array([1.5, -2.0, 3.0, 4.0], dtype=np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
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

    # Input 2: 2D int32 array, axis [1], keepdims True
    a = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    axis = [1]
    dtype = np.int32
    keepdims = True
    initial = np.array(2, dtype=np.int32)
    where = np.array([[True, True, True], [False, True, True]], dtype=bool)
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

    # Input 3: 2D float32 array, axis [0], keepdims False
    a = np.random.randn(4, 5).astype(np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((4, 5), dtype=bool)
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

    # Input 4: 3D int32 array, axis [0, 2], keepdims True
    a = np.array(
        [[[1, 2], [3, 4]], [[-1, 2], [-3, 1]], [[2, 1], [1, -2]]],
        dtype=np.int32,
    )
    axis = [0, 2]
    dtype = np.int32
    keepdims = True
    initial = np.array(1, dtype=np.int32)
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

    # Input 5: 3D float32 array, axis [-1], broadcasted where
    a = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    axis = [-1]
    dtype = np.float32
    keepdims = False
    initial = np.array(0.5, dtype=np.float32)
    where = np.array([[[True], [False], [True]]], dtype=bool)
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

    # Input 6: 4D float32 array, axis [1, 3], keepdims True
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = [1, 3]
    dtype = np.float32
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 3, 2, 4), dtype=bool)
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

    # Input 7: 1D int32 array, axis [0], keepdims False
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    axis = [0]
    dtype = np.int32
    keepdims = False
    initial = np.array(1, dtype=np.int32)
    where = np.array([True, True, True, False, True], dtype=bool)
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

    # Input 8: 2D float32 array, axis [0], keepdims True
    a = np.array([[1.5, 2.5], [3.0, -1.0]], dtype=np.float32)
    axis = [0]
    dtype = np.float32
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
    where = np.array([[True, False], [True, True]], dtype=bool)
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

    # Input 9: 3D int32 array, axis [-2, -1], keepdims False
    a = np.random.randint(-3, 4, size=(4, 2, 3)).astype(np.int32)
    axis = [-2, -1]
    dtype = np.int32
    keepdims = False
    initial = np.array(1, dtype=np.int32)
    where = a != 0
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

    # Input 10: 2D float32 array, axis [0, 1], reduce all specified axes
    a = np.array([[0.1, -0.5, 2.0], [1.5, 3.0, -1.0]], dtype=np.float32)
    axis = [0, 1]
    dtype = np.float32
    keepdims = False
    initial = np.array(2.0, dtype=np.float32)
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

    return list_of_inputs


generated_inputs["jax.numpy.prod_3"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_3'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_3'], lib="jax", suffix=3)
