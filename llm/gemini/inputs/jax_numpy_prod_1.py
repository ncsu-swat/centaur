
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    a = np.array([1.5, -2.0, 3.0, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.dtype(np.float32)
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

    # Input 2: 2D int32 array, axis 1, keepdims=True
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = 1
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(2, dtype=np.int32)
    where = np.array([[True, False, True], [True, True, True]], dtype=bool)
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

    # Input 3: 2D float32 array, axis -1
    a = np.array([[-1.0, 2.5], [3.0, -4.0]], dtype=np.float32)
    axis = -1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
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

    # Input 4: 3D float32 array, axis 0
    a = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    axis = 0
    dtype = np.dtype(np.float32)
    keepdims = True
    initial = np.array(0.5, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
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

    # Input 5: 3D int32 array, axis 2
    a = np.arange(1, 25, dtype=np.int32).reshape((2, 3, 4))
    axis = 2
    dtype = np.dtype(np.int32)
    keepdims = False
    initial = np.array(1, dtype=np.int32)
    where = a % 2 == 1
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

    # Input 6: 1D int32 array with negative elements
    a = np.array([-2, 3, -4, 5], dtype=np.int32)
    axis = 0
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.array([True, True, True, False], dtype=bool)
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

    # Input 7: 4D float32 array, axis 1
    a = np.random.uniform(0.5, 1.5, size=(2, 2, 2, 2)).astype(np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(1.0, dtype=np.float32)
    where = np.ones((2, 2, 2, 2), dtype=bool)
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

    # Input 8: 2D int32 array with broadcastable where
    a = np.array([[2, 3], [4, 5]], dtype=np.int32)
    axis = 0
    dtype = np.dtype(np.int32)
    keepdims = True
    initial = np.array(1, dtype=np.int32)
    where = np.array([[True, False]], dtype=bool)
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

    # Input 9: 2D float32 with negative axis -2
    a = np.array([[0.5, -1.5, 2.0], [3.0, -0.5, 1.0]], dtype=np.float32)
    axis = -2
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(2.0, dtype=np.float32)
    where = np.array([[True, True, True], [False, True, True]], dtype=bool)
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

    # Input 10: 2D float32 array, all where elements False
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis = 1
    dtype = np.dtype(np.float32)
    keepdims = False
    initial = np.array(3.0, dtype=np.float32)
    where = np.zeros((2, 2), dtype=bool)
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


generated_inputs["jax.numpy.prod_1"] = prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.prod_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.prod_1'.")


check_valid('jax.numpy.prod', generated_inputs['jax.numpy.prod_1'], lib="jax", suffix=1)
