
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, delete single element
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    obj = np.array([1], dtype=np.int32)
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

    # Input 2: 2D int32 array, delete multiple rows along axis 0
    arr = np.arange(12, dtype=np.int32).reshape(4, 3)
    obj = np.array([0, 2], dtype=np.int32)
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

    # Input 3: 2D float64 array, delete columns along axis 1
    arr = np.random.randn(3, 5).astype(np.float64)
    obj = np.array([1, 3], dtype=np.int32)
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

    # Input 4: 3D complex64 array, delete slices along axis 1
    arr = (np.random.randn(2, 4, 3) + 1j * np.random.randn(2, 4, 3)).astype(
        np.complex64
    )
    obj = np.array([0, 2], dtype=np.int32)
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

    # Input 5: 1D boolean array, delete elements
    arr = np.array([True, False, True, True, False, True], dtype=bool)
    obj = np.array([2, 5], dtype=np.int32)
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

    # Input 6: 4D int64 array, negative axis
    arr = np.ones((2, 3, 4, 5), dtype=np.int64)
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

    # Input 7: 2D float32 array, negative axis
    arr = np.random.randn(6, 4).astype(np.float32)
    obj = np.array([1, 4], dtype=np.int32)
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

    # Input 8: 3D uint8 array, delete along axis 2
    arr = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
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

    # Input 9: 1D float64 array, delete multiple elements
    arr = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    obj = np.array([0, 2, 4, 6], dtype=np.int32)
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

    # Input 10: 3D int16 array, delete single element along axis 0
    arr = np.zeros((3, 2, 2), dtype=np.int16)
    obj = np.array([0], dtype=np.int32)
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


generated_inputs["jax.numpy.delete_3"] = delete_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.delete_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.delete_3'.")


check_valid('jax.numpy.delete', generated_inputs['jax.numpy.delete_3'], lib="jax", suffix=3)
