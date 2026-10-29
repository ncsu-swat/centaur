
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def insert_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, single scalar index, scalar value, axis=0
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    obj = np.array(2, dtype=np.int64)
    values = np.array(99, dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 2: 1D float array, multiple indices, matching 1D values, axis=0
    arr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.array([10.5, 20.5], dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 3: 1D int array, negative axis=-1
    arr = np.array([10, 20, 30, 40], dtype=np.int64)
    obj = np.array([0, 2], dtype=np.int32)
    values = np.array([100, 200], dtype=np.int64)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 4: 2D float32 array, insert rows along axis=0 with matching shape (1, 3)
    arr = np.ones((2, 3), dtype=np.float32)
    obj = np.array([1], dtype=np.int32)
    values = np.zeros((1, 3), dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 5: 2D int32 array, insert columns along axis=1 with matching shape (2, 2)
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.array([[10, 11], [12, 13]], dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 6: 2D float64 array, scalar obj, scalar value broadcasting along axis=1
    arr = np.arange(12, dtype=np.float64).reshape((3, 4))
    obj = np.array(2, dtype=np.int32)
    values = np.array(-1.0, dtype=np.float64)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 7: 2D int array, negative axis=-1 with broadcasted scalar value
    arr = np.array([[5, 6], [7, 8]], dtype=np.int32)
    obj = np.array([0, 1], dtype=np.int32)
    values = np.array(0, dtype=np.int32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 8: 3D float32 array, insert along axis=0 with shape (1, 2, 2)
    arr = np.ones((3, 2, 2), dtype=np.float32)
    obj = np.array([0], dtype=np.int32)
    values = np.zeros((1, 2, 2), dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 9: 3D int64 array, scalar obj, scalar values, axis=2
    arr = np.arange(24, dtype=np.int64).reshape((2, 3, 4))
    obj = np.array(1, dtype=np.int64)
    values = np.array(-99, dtype=np.int64)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 10: 3D float32 array, negative axis=-2, scalar value
    arr = np.random.randn(2, 4, 3).astype(np.float32)
    obj = np.array([1, 2], dtype=np.int32)
    values = np.array(3.14, dtype=np.float32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 11: 4D float32 array, insert along axis=0 with matching shape
    arr = np.ones((2, 2, 2, 2), dtype=np.float32)
    obj = np.array([1], dtype=np.int32)
    values = np.zeros((1, 2, 2, 2), dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 12: 1D bool array, inserting boolean values
    arr = np.array([True, False, True], dtype=np.bool_)
    obj = np.array([1, 2], dtype=np.int32)
    values = np.array([False, False], dtype=np.bool_)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.insert_2"] = insert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.insert_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.insert_2'.")


check_valid('jax.numpy.insert', generated_inputs['jax.numpy.insert_2'], lib="jax", suffix=2)
