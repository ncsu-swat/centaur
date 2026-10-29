
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ravel_multi_index_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D indexing, C order, default raise mode
    multi_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
    dims = (3, 4)
    mode = "raise"
    order = "C"
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 2: 2D indexing, Fortran order
    multi_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    dims = (3, 4)
    mode = "raise"
    order = "F"
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 3: 3D indexing, clip mode with out-of-bound indices
    multi_index = np.array(
        [
            [-1, 0, 5],
            [1, 10, 2],
            [0, 2, -3],
        ],
        dtype=np.int32,
    )
    dims = (3, 4, 5)
    mode = "clip"
    order = "C"
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 4: 3D indexing, wrap mode
    multi_index = np.array(
        [
            [4, -1, 2],
            [-2, 5, 1],
            [6, -4, 3],
        ],
        dtype=np.int32,
    )
    dims = (3, 4, 5)
    mode = "wrap"
    order = "C"
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 5: 1D indexing, ignore mode
    multi_index = np.array([[0, 2, 4, 1]], dtype=np.int32)
    dims = (10,)
    mode = "ignore"
    order = "C"
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 6: 4D indexing, 3D array (shape: (4, 2, 3))
    multi_index = np.zeros((4, 2, 3), dtype=np.int32)
    dims = (2, 3, 4, 5)
    mode = "raise"
    order = "C"
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 7: 4D indexing with F order and clip mode
    multi_index = np.array(
        [
            [[0, 1], [1, 0]],
            [[2, 3], [0, 1]],
            [[1, 2], [3, 0]],
            [[0, 1], [2, 3]],
        ],
        dtype=np.int64,
    )
    dims = (2, 4, 4, 4)
    mode = "clip"
    order = "F"
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 8: Scalar indexing along 2 dimensions (shape: (2,))
    multi_index = np.array([2, 3], dtype=np.int32)
    dims = (5, 5)
    mode = "raise"
    order = "C"
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 9: 5D indexing, empty trailing dimension (shape: (5, 0))
    multi_index = np.empty((5, 0), dtype=np.int32)
    dims = (2, 3, 2, 4, 2)
    mode = "raise"
    order = "C"
    dtype = np.dtype(np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    # Input 10: 2D indexing with wrap mode and F order
    multi_index = np.array([[-5, 12, 3], [8, -7, 2]], dtype=np.int64)
    dims = (6, 5)
    mode = "wrap"
    order = "F"
    dtype = np.dtype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "multi_index": multi_index,
                "dims": dims,
                "mode": mode,
                "order": order,
                "dtype": dtype,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.ravel_multi_index_1"] = ravel_multi_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel_multi_index_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel_multi_index_1'.")


check_valid('jax.numpy.ravel_multi_index', generated_inputs['jax.numpy.ravel_multi_index_1'], lib="jax", suffix=1)
