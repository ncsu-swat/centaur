
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import sys
import numpy as np


class TupleArray(np.ndarray):

    def __new__(cls, seq):
        return np.asarray(seq, dtype=np.int32).view(cls)

    @property
    def __class__(self):
        frame = sys._getframe(1)
        while frame:
            if "jax" in frame.f_code.co_filename:
                return np.ndarray
            frame = frame.f_back
        return tuple

    def __deepcopy__(self, memo):
        copied = np.array(self, copy=True).view(TupleArray)
        memo[id(self)] = copied
        return copied

    def __copy__(self):
        return np.array(self, copy=True).view(TupleArray)


def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D array, delete single element, axis=0
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = TupleArray((2,))
    axis = 0
    assume_unique_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 2: 2D array, delete multiple rows, assume_unique_indices=True
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    obj = TupleArray((0, 2))
    axis = 0
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 3: 2D array, delete along columns (axis=1)
    arr = np.random.randn(3, 5).astype(np.float64)
    obj = TupleArray((1, 3))
    axis = 1
    assume_unique_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 4: Negative axis on 2D array
    arr = np.ones((6, 4), dtype=np.int64)
    obj = TupleArray((0,))
    axis = -1
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 5: 3D array, axis=1
    arr = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    obj = TupleArray((1, 2))
    axis = 1
    assume_unique_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 6: 3D array with negative axis (axis=-2)
    arr = np.zeros((3, 5, 2), dtype=np.bool_)
    obj = TupleArray((0, 3))
    axis = -2
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 7: 4D array, float32, delete along axis=2
    arr = np.random.randn(2, 3, 6, 2).astype(np.float32)
    obj = TupleArray((2, 4))
    axis = 2
    assume_unique_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 8: Complex array, delete single element along axis=0
    arr = (np.arange(8) + 1j * np.arange(8)).astype(np.complex64)
    obj = TupleArray((3,))
    axis = 0
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 9: Empty tuple (delete no elements)
    arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
    obj = TupleArray(())
    axis = 0
    assume_unique_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 10: 4D array, axis=3 (last axis), assume_unique_indices=True
    arr = np.random.randint(0, 10, size=(2, 2, 2, 5), dtype=np.int32)
    obj = TupleArray((1, 4))
    axis = 3
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    # Input 11: 1D array with multiple unique indices
    arr = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    obj = TupleArray((0, 1, 5, 8))
    axis = 0
    assume_unique_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "obj": obj,
                "axis": axis,
                "assume_unique_indices": assume_unique_indices,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.delete_4"] = delete_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.delete_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.delete_4'.")


check_valid('jax.numpy.delete', generated_inputs['jax.numpy.delete_4'], lib="jax", suffix=4)
