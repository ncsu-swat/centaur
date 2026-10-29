
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 2: 2D array, axis 1, keepdims=True, return_sign=True
    a = np.array([[1.5, 2.5, -1.0], [-0.5, 0.0, 3.2]], dtype=np.float32)
    axis = [1]
    b = np.ones((2, 3), dtype=np.float32)
    keepdims = True
    return_sign = True
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 3: 2D float64 with negative scaling factors
    a = np.array([[-1.0, 2.5], [3.0, -0.5]], dtype=np.float64)
    axis = [0]
    b = np.array([[1.0, -1.0], [2.0, 1.0]], dtype=np.float64)
    keepdims = False
    return_sign = True
    where = np.array([[True, True], [True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 4: 3D array reducing over multiple axes
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.1
    axis = [0, 2]
    b = np.ones((2, 3, 4), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 5: 1D array with negative numbers and selective masking
    a = np.array([-10.0, -20.0, -5.0, -1.0], dtype=np.float32)
    axis = [0]
    b = np.array([0.5, 1.5, 2.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, False, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 6: 4D array with multiple axes reduction
    a = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    axis = [1, 3]
    b = np.ones((2, 2, 2, 2), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.ones((2, 2, 2, 2), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 7: 2D array with broadcasted scaling factors b
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    axis = [0]
    b = np.array([[0.5, 1.0, 1.5]], dtype=np.float64)
    keepdims = True
    return_sign = True
    where = np.ones((2, 3), dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 8: 3D array with negative axis indexing
    a = np.arange(12, dtype=np.float32).reshape(2, 2, 3) - 6.0
    axis = [-1]
    b = np.ones((2, 2, 3), dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.array(
        [
            [[True, False, True], [True, True, False]],
            [[False, True, True], [True, False, True]],
        ],
        dtype=bool,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 9: 2D array reduction over all axes using list of axes
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    axis = [0, 1]
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 10: Numerically large values
    a = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.array([True, True, True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    # Input 11: Float16 array
    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    axis = [1]
    b = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float16)
    keepdims = False
    return_sign = False
    where = np.array([[True, True], [False, True]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "b": b,
                "keepdims": keepdims,
                "return_sign": return_sign,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.nn.logsumexp_3"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logsumexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logsumexp_3'.")


check_valid('jax.nn.logsumexp', generated_inputs['jax.nn.logsumexp_3'], lib="jax", suffix=3)
