
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def insert_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array insertion of a single value
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    obj = np.array([2], dtype=np.int32)
    values = np.array([99], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 2: 2D array inserting columns
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.array([[10, 11], [12, 13]], dtype=np.int32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 3: 1D float array with negative index
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    obj = np.array([-1], dtype=np.int32)
    values = np.array([15.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 4: 2D array inserting rows
    arr = np.random.randn(4, 4).astype(np.float64)
    obj = np.array([1, 3], dtype=np.int32)
    values = np.random.randn(2, 4).astype(np.float64)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 5: 3D array inserting along axis 2 with broadcasting values
    arr = np.random.randn(3, 3, 3).astype(np.float32)
    obj = np.array([2], dtype=np.int32)
    values = np.array([0.0], dtype=np.float32)
    axis = 2
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 6: 1D array multiple insertion at same index
    arr = np.array([0, 1, 2], dtype=np.float32)
    obj = np.array([0, 0, 0], dtype=np.int32)
    values = np.array([9.0, 8.0, 7.0], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 7: 1D array int64 type
    arr = np.arange(10, dtype=np.int64)
    obj = np.array([0, 2, 4, 6], dtype=np.int32)
    values = np.array([100, 200, 300, 400], dtype=np.int64)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 8: 3D array inserting a sub-tensor
    arr = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    obj = np.array([1], dtype=np.int32)
    values = np.ones((2, 1, 2), dtype=np.int32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 9: 2D array inserting multiple rows with negative index
    arr = np.random.randn(3, 2).astype(np.float32)
    obj = np.array([-2], dtype=np.int32)
    values = np.random.randn(1, 2).astype(np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 10: 1D array inserting at the final boundary
    arr = np.array([10, 20, 30], dtype=np.int32)
    obj = np.array([3], dtype=np.int32)
    values = np.array([40], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

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
