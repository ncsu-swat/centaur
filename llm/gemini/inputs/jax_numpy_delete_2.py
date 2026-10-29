
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 1D integer indices, axis 0, False
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = np.array([1, 3], dtype=np.int32)
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

    # Input 2: 1D array, 1D unique indices, axis 0, True
    arr = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    obj = np.array([0, 2, 4], dtype=np.int32)
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

    # Input 3: 2D array, delete rows (axis=0)
    arr = np.arange(12, dtype=np.float64).reshape(4, 3)
    obj = np.array([0, 2], dtype=np.int64)
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

    # Input 4: 2D array, delete columns (axis=1)
    arr = np.arange(15, dtype=np.int64).reshape(3, 5)
    obj = np.array([1, 4], dtype=np.int32)
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

    # Input 5: 2D array with negative axis and negative indices
    arr = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32
    )
    obj = np.array([-1, -3], dtype=np.int32)
    axis = -2
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

    # Input 6: 3D array, axis=2 (last axis)
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    obj = np.array([0, 3], dtype=np.int32)
    axis = 2
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

    # Input 7: 3D array, axis=1 (middle axis)
    arr = np.random.randint(0, 100, size=(3, 6, 2), dtype=np.int32)
    obj = np.array([1, 2, 4], dtype=np.int32)
    axis = 1
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

    # Input 8: 1D boolean array for obj (mask)
    arr = np.array([100, 200, 300, 400], dtype=np.int64)
    obj = np.array([True, False, True, False], dtype=np.bool_)
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

    # Input 9: 2D array, boolean mask along axis=1
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    obj = np.array([False, True, False, True], dtype=np.bool_)
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

    # Input 10: 4D array, axis=-1
    arr = np.ones((2, 2, 3, 5), dtype=np.complex64)
    obj = np.array([2], dtype=np.int32)
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

    # Input 11: Single element deletion via 0D/1D scalar tensor
    arr = np.array([7, 8, 9, 10], dtype=np.int16)
    obj = np.array(2, dtype=np.int32)
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

    # Input 12: Empty deletion indices
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    obj = np.array([], dtype=np.int32)
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


generated_inputs["jax.numpy.delete_2"] = delete_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.delete_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.delete_2'.")


check_valid('jax.numpy.delete', generated_inputs['jax.numpy.delete_2'], lib="jax", suffix=2)
