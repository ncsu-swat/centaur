
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def insert_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, inserting a single element at index 2
    arr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    obj = np.array([2], dtype=np.int32)
    values = np.array([99], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 2: 1D float32 array, inserting multiple elements
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.array([10.5, 20.5], dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 3: 2D int32 array, inserting columns (axis=1)
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.array([[10, 11], [12, 13]], dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 4: 2D float64 array, inserting rows (axis=0)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    obj = np.array([0, 2], dtype=np.int32)
    values = np.ones((2, 4), dtype=np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 5: 2D array, inserting with negative axis (axis=-1)
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    obj = np.array([1], dtype=np.int32)
    values = np.array([[100], [200]], dtype=np.int64)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 6: 3D float32 array, inserting along axis 0
    arr = np.ones((2, 3, 4), dtype=np.float32)
    obj = np.array([1], dtype=np.int32)
    values = np.zeros((1, 3, 4), dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 7: 3D int32 array, inserting along axis 1
    arr = np.arange(24, dtype=np.int32).reshape(2, 4, 3)
    obj = np.array([0, 3], dtype=np.int32)
    values = np.full((2, 2, 3), 7, dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 8: 3D float64 array, inserting along axis 2
    arr = np.zeros((3, 2, 2), dtype=np.float64)
    obj = np.array([1, 2], dtype=np.int32)
    values = np.ones((3, 2, 2), dtype=np.float64)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 9: 1D array, inserting at negative indices
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = np.array([-2, -1], dtype=np.int32)
    values = np.array([999, 888], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 10: 4D float32 array, inserting along negative axis (axis=-2)
    arr = np.random.randn(2, 2, 3, 2).astype(np.float32)
    obj = np.array([0], dtype=np.int32)
    values = np.random.randn(2, 2, 1, 2).astype(np.float32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 11: 2D bool array, inserting columns
    arr = np.array([[True, False], [False, True]], dtype=np.bool_)
    obj = np.array([1], dtype=np.int32)
    values = np.array([[True], [False]], dtype=np.bool_)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.insert_4"] = insert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.insert_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.insert_4'.")


check_valid('jax.numpy.insert', generated_inputs['jax.numpy.insert_4'], lib="jax", suffix=4)
