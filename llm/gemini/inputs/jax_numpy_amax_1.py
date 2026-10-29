
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def amax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, keepdims False
    a = np.array([1.5, -2.3, 3.7, -4.1, 0.0], dtype=np.float32)
    axis = 0
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.array([True, True, True, False, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 2: 2D float64 array, axis 1, keepdims True
    a = np.random.randn(3, 4).astype(np.float64)
    axis = 1
    keepdims = True
    initial = np.array(-np.inf, dtype=np.float64)
    where = np.ones((3, 4), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 3: 2D int32 array, axis 0, keepdims False
    a = np.array([[-10, 20, 3], [45, -5, -60]], dtype=np.int32)
    axis = 0
    keepdims = False
    initial = np.array(-100, dtype=np.int32)
    where = np.array([[True, False, True], [True, True, False]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 4: 3D float32 array, axis -1, keepdims True
    a = np.random.randn(2, 3, 5).astype(np.float32)
    axis = -1
    keepdims = True
    initial = np.array(-1000.0, dtype=np.float32)
    where = (a > 0).astype(np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 5: 3D int64 array, axis 1, keepdims False
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 12
    axis = 1
    keepdims = False
    initial = np.array(-50, dtype=np.int64)
    where = np.ones((2, 3, 4), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 6: 4D float32 array, axis 2, keepdims False
    a = np.random.uniform(-5.0, 5.0, size=(2, 2, 4, 3)).astype(np.float32)
    axis = 2
    keepdims = False
    initial = np.array(-5.0, dtype=np.float32)
    where = np.ones((2, 2, 4, 3), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 7: 1D int16 array, axis -1, keepdims True
    a = np.array([-300, -200, -500, -100], dtype=np.int16)
    axis = -1
    keepdims = True
    initial = np.array(-1000, dtype=np.int16)
    where = np.array([True, True, False, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 8: 2D uint8 array, axis 0, keepdims True
    a = np.array([[10, 250], [50, 20]], dtype=np.uint8)
    axis = 0
    keepdims = True
    initial = np.array(0, dtype=np.uint8)
    where = np.array([[True, False], [True, True]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 9: 3D float64 array, axis -2, keepdims False
    a = np.linspace(-10, 10, 18, dtype=np.float64).reshape((3, 3, 2))
    axis = -2
    keepdims = False
    initial = np.array(-20.0, dtype=np.float64)
    where = np.ones((3, 3, 2), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    # Input 10: 2D boolean array, axis 1, keepdims False
    a = np.array([[False, True, False], [False, False, False]], dtype=np.bool_)
    axis = 1
    keepdims = False
    initial = np.array(False, dtype=np.bool_)
    where = np.ones((2, 3), dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "axis": axis,
                "keepdims": keepdims,
                "initial": initial,
                "where": where,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.amax_1"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amax_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amax_1'.")


check_valid('jax.numpy.amax', generated_inputs['jax.numpy.amax_1'], lib="jax", suffix=1)
