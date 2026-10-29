
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class ShapedList(list):

    @property
    def shape(self):
        if len(self) > 0 and hasattr(self[0], "shape"):
            return (len(self),) + getattr(self[0], "shape", ())
        return (len(self),)

    @property
    def dtype(self):
        if len(self) > 0 and hasattr(self[0], "dtype"):
            return self[0].dtype
        return np.dtype(np.int32)


def ravel_multi_index_inputs():
    list_of_inputs = []

    # Input 1: 2D basic coordinates, mode='raise', order='C'
    multi_index = ShapedList(
        [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([1, 2, 0], dtype=np.int32),
        ]
    )
    dims = ShapedList([3, 4])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D coordinates with out-of-bound indices, mode='wrap', order='C'
    multi_index = ShapedList(
        [
            np.array([-1, 0, 5], dtype=np.int64),
            np.array([2, -2, 6], dtype=np.int64),
        ]
    )
    dims = ShapedList([4, 4])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "wrap",
        "order": "C",
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D coordinates with out-of-bound indices, mode='clip', order='F'
    multi_index = ShapedList(
        [
            np.array([-3, 1, 10], dtype=np.int32),
            np.array([0, 8, -1], dtype=np.int32),
        ]
    )
    dims = ShapedList([5, 6])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "clip",
        "order": "F",
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D index
    multi_index = ShapedList(
        [
            np.array([0, 3, 2, 1], dtype=np.int64),
        ]
    )
    dims = ShapedList([10])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D coordinates, scalar (0D) arrays
    multi_index = ShapedList(
        [
            np.array(2, dtype=np.int32),
            np.array(1, dtype=np.int32),
            np.array(0, dtype=np.int32),
        ]
    )
    dims = ShapedList([3, 4, 2])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "F",
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D multi_index with 2D array inputs
    multi_index = ShapedList(
        [
            np.array([[0, 1], [1, 2]], dtype=np.int32),
            np.array([[1, 0], [2, 1]], dtype=np.int32),
            np.array([[0, 1], [0, 0]], dtype=np.int32),
        ]
    )
    dims = ShapedList([3, 3, 2])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D multi_index 1D arrays
    multi_index = ShapedList(
        [
            np.array([0, 1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([2, 0], dtype=np.int64),
        ]
    )
    dims = ShapedList([2, 3, 2, 4])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "C",
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D broadcasting shapes (3, 1) and (1, 4)
    multi_index = ShapedList(
        [
            np.array([[0], [1], [2]], dtype=np.int32),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
        ]
    )
    dims = ShapedList([3, 5])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "raise",
        "order": "F",
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D negative values with wrap mode
    multi_index = ShapedList(
        [
            np.array([-2, -1, 0], dtype=np.int64),
            np.array([-3, 1, -1], dtype=np.int64),
            np.array([0, -2, 1], dtype=np.int64),
        ]
    )
    dims = ShapedList([4, 5, 3])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "wrap",
        "order": "C",
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D multi_index with large indices and clip mode
    multi_index = ShapedList(
        [
            np.array([100, -50, 2], dtype=np.int32),
            np.array([-10, 200, 3], dtype=np.int32),
        ]
    )
    dims = ShapedList([10, 20])
    input_dict = {
        "multi_index": multi_index,
        "dims": dims,
        "mode": "clip",
        "order": "C",
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ravel_multi_index_2"] = ravel_multi_index_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel_multi_index_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel_multi_index_2'.")


check_valid('jax.numpy.ravel_multi_index', generated_inputs['jax.numpy.ravel_multi_index_2'], lib="jax", suffix=2)
