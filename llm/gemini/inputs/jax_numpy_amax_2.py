
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def amax_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, full reduction along axis 0
    a = np.array([-3.5, 1.2, 7.8, -0.4, 2.1], dtype=np.float32)
    axis = (0,)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.array([True, True, True, True, True], dtype=bool)
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

    # Input 2: 2D float64 array, reduce along axis 1 with keepdims=True and mask
    a = np.array([[1.0, -5.0, 3.0], [-2.0, 4.0, -1.0]], dtype=np.float64)
    axis = (1,)
    keepdims = True
    initial = np.array(-np.inf, dtype=np.float64)
    where = np.array(
        [[True, False, True], [False, True, True]], dtype=bool
    )
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

    # Input 3: 2D int32 array, reduce along axis 0
    a = np.array([[10, -20, 30], [40, 5, -15]], dtype=np.int32)
    axis = (0,)
    keepdims = False
    initial = np.array(-1000, dtype=np.int32)
    where = np.ones((2, 3), dtype=bool)
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

    # Input 4: 3D float32 array, reduction over multiple axes (0, 2)
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 12.0
    axis = (0, 2)
    keepdims = True
    initial = np.array(-50.0, dtype=np.float32)
    where = np.ones((2, 3, 4), dtype=bool)
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

    # Input 5: 3D int64 array with non-trivial where mask
    a = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[-9, -10], [-11, -12]]],
        dtype=np.int64,
    )
    axis = (1,)
    keepdims = False
    initial = np.array(-999999, dtype=np.int64)
    where = np.array(
        [
            [[True, False], [True, True]],
            [[False, True], [True, False]],
            [[True, True], [False, True]],
        ],
        dtype=bool,
    )
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

    # Input 6: 4D float32 array, multi-axis reduction (1, 3)
    a = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = (1, 3)
    keepdims = False
    initial = np.array(-10.0, dtype=np.float32)
    where = np.ones((2, 3, 2, 4), dtype=bool)
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

    # Input 7: 1D uint8 array with keepdims=True
    a = np.array([10, 255, 0, 128], dtype=np.uint8)
    axis = (0,)
    keepdims = True
    initial = np.array(0, dtype=np.uint8)
    where = np.array([True, False, True, True], dtype=bool)
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

    # Input 8: 2D float32 array, all negative values, reduction over all axes
    a = np.array([[-10.5, -20.2], [-30.1, -40.9]], dtype=np.float32)
    axis = (0, 1)
    keepdims = False
    initial = np.array(-100.0, dtype=np.float32)
    where = np.ones((2, 2), dtype=bool)
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

    # Input 9: 3D int16 array, reducing over axes (0, 1, 2) with keepdims=True
    a = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    axis = (0, 1, 2)
    keepdims = True
    initial = np.array(-32768, dtype=np.int16)
    where = np.array(
        [[[True, False], [True, True]], [[False, False], [True, False]]],
        dtype=bool,
    )
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

    # Input 10: 2D float64 array, reduce along axis 0 with broadcasted where mask
    a = np.linspace(-5.0, 5.0, 12, dtype=np.float64).reshape(4, 3)
    axis = (0,)
    keepdims = True
    initial = np.array(-1e6, dtype=np.float64)
    where = np.array([[True, False, True]], dtype=bool)
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


generated_inputs["jax.numpy.amax_2"] = amax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.amax_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.amax_2'.")


check_valid('jax.numpy.amax', generated_inputs['jax.numpy.amax_2'], lib="jax", suffix=2)
