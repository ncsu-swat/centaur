
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def insert_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, insert at beginning along axis 0
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    obj = 0
    values = np.array(3.14, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 2: 1D int32 array, insert at index 2 along axis 0
    arr = np.array([10, 20, 30, 40], dtype=np.int32)
    obj = 2
    values = np.array(99, dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 3: 1D int64 array, insert before last element using negative index
    arr = np.array([100, 200, 300, 400], dtype=np.int64)
    obj = -1
    values = np.array(-5, dtype=np.int64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 4: 2D float32 array, insert a row at index 1 along axis 0
    arr = np.ones((3, 4), dtype=np.float32)
    obj = 1
    values = np.array(0.0, dtype=np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 5: 2D float64 array, insert a column at index 0 along axis 1
    arr = np.ones((3, 4), dtype=np.float64)
    obj = 0
    values = np.array(1.5, dtype=np.float64)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 6: 2D int32 array, insert a column before the last column using negative index
    arr = np.arange(12, dtype=np.int32).reshape((3, 4))
    obj = -1
    values = np.array(42, dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 7: 3D float32 array, insert along axis 1
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    obj = 1
    values = np.array(-1.0, dtype=np.float32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 8: 3D int32 array, insert along axis 2
    arr = np.zeros((2, 3, 4), dtype=np.int32)
    obj = 2
    values = np.array(7, dtype=np.int32)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 9: 3D float64 array, insert along negative axis -3
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    obj = 0
    values = np.array(2.718, dtype=np.float64)
    axis = -3
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 10: 4D float32 array, insert along last axis -1
    arr = np.ones((2, 2, 3, 4), dtype=np.float32)
    obj = 1
    values = np.array(0.5, dtype=np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 11: 1D boolean array, insert boolean value
    arr = np.array([True, False, True], dtype=bool)
    obj = 1
    values = np.array(False, dtype=bool)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    # Input 12: 2D complex64 array, insert complex value row along axis 0
    arr = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    obj = 1
    values = np.array(1 + 2j, dtype=np.complex64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "obj": obj, "values": values, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.insert_1"] = insert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.insert_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.insert_1'.")


check_valid('jax.numpy.insert', generated_inputs['jax.numpy.insert_1'], lib="jax", suffix=1)
