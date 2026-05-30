
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def insert_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D insertion, int32
    arr = np.array([1, 2, 3], dtype=np.int32)
    obj = 1
    values = np.array([99], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 2: 1D insertion with float32 and multiple values
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    obj = 2
    values = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 3: 2D insertion of a row (axis 0)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    obj = 1
    values = np.array([[10, 11, 12]], dtype=np.int32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 4: 2D insertion of multiple rows (axis 0)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    obj = 0
    values = np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 5: 2D insertion of a column (axis 1)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    obj = 2
    values = np.array([[10], [20]], dtype=np.int32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 6: 2D insertion with negative axis and negative index
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    obj = -1
    values = np.array([[10], [20]], dtype=np.int32)
    axis = -1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 7: 3D insertion, axis 1
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    obj = 1
    values = np.ones((2, 1, 4), dtype=np.int32)
    axis = 1
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 8: 3D float64 insertion, negative index, axis 2
    arr = np.random.randn(2, 2, 3).astype(np.float64)
    obj = -1
    values = np.random.randn(2, 2, 2).astype(np.float64)
    axis = 2
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 9: Boolean arrays
    arr = np.array([[True, False], [False, True]], dtype=bool)
    obj = 1
    values = np.array([[True, True]], dtype=bool)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

    # Input 10: 1D array with int64 and negative index
    arr = np.arange(5, dtype=np.int64)
    obj = -1
    values = np.array([9, 8], dtype=np.int64)
    axis = 0
    list_of_inputs.append({"arr": arr, "obj": obj, "values": values, "axis": axis})

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
