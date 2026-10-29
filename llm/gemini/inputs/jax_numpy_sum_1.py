
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sum_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, no keepdims
    a = np.array([1.0, -2.0, 3.5, 4.0], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = np.array([True, True, False, True], dtype=bool)
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

    # Input 2: 2D float32 array, axis 1, keepdims True
    a = np.array([[-1.5, 2.0, 3.2], [4.1, -5.0, 6.7]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    keepdims = True
    initial = np.array(10.0, dtype=np.float32)
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

    # Input 3: 3D int32 array, axis -1, keepdims False
    a = np.arange(24, dtype=np.int32).reshape((2, 3, 4)) - 10
    axis = -1
    dtype = np.int32
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = a > 0
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

    # Input 4: 2D int32 array, axis 0, keepdims True
    a = np.array([[10, -20], [30, 40], [-50, 60]], dtype=np.int32)
    axis = 0
    dtype = np.int32
    keepdims = True
    initial = np.array(-5, dtype=np.int32)
    where = np.array([[True, False], [True, True], [False, True]], dtype=bool)
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

    # Input 5: 4D float32 array, axis 2, keepdims False
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = 2
    dtype = np.float32
    keepdims = False
    initial = np.array(0.0, dtype=np.float32)
    where = a > -0.5
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

    # Input 6: 1D bool array, axis 0, cast to int32
    a = np.array([True, False, True, True, False], dtype=bool)
    axis = 0
    dtype = np.int32
    keepdims = False
    initial = np.array(0, dtype=np.int32)
    where = np.array([True, True, True, False, False], dtype=bool)
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

    # Input 7: 2D float32 array, axis 0, keepdims True
    a = np.array([[1.0, -3.0], [5.0, 0.0]], dtype=np.float32)
    axis = 0
    dtype = np.float32
    keepdims = True
    initial = np.array(1.0, dtype=np.float32)
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

    # Input 8: 3D float32 array, negative axis -2, keepdims True
    a = np.ones((3, 4, 2), dtype=np.float32) * -1.5
    axis = -2
    dtype = np.float32
    keepdims = True
    initial = np.array(0.0, dtype=np.float32)
    where = np.ones((3, 4, 2), dtype=bool)
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

    # Input 9: 2D float32 array with broadcasted where, axis 1, keepdims False
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = 1
    dtype = np.float32
    keepdims = False
    initial = np.array(-1.0, dtype=np.float32)
    where = np.array([[True], [False]], dtype=bool)
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

    # Input 10: 1D int32 array, axis 0, keepdims True
    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    axis = 0
    dtype = np.int32
    keepdims = True
    initial = np.array(100, dtype=np.int32)
    where = np.array([True, False, True, False, True], dtype=bool)
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


generated_inputs["jax.numpy.sum_1"] = sum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sum_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sum_1'.")


check_valid('jax.numpy.sum', generated_inputs['jax.numpy.sum_1'], lib="jax", suffix=1)
