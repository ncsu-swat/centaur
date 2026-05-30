
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D array, delete single element
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = np.array([2], dtype=np.int32)
    axis = 0
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, delete multiple elements, unique
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    obj = np.array([0, 4], dtype=np.int32)
    axis = 0
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, delete a row
    arr = np.random.randn(4, 3).astype(np.float32)
    obj = np.array([1], dtype=np.int32)
    axis = 0
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, delete multiple rows
    arr = np.random.randn(5, 3).astype(np.float32)
    obj = np.array([0, 2, 4], dtype=np.int32)
    axis = 0
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, delete col with float64
    arr = np.random.randn(3, 4).astype(np.float64)
    obj = np.array([2], dtype=np.int32)
    axis = 1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, delete multiple columns
    arr = np.random.randn(3, 5).astype(np.float32)
    obj = np.array([1, 3], dtype=np.int32)
    axis = 1
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, delete along axis 2
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    obj = np.array([0, 2, 3], dtype=np.int32)
    axis = 2
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, delete single along axis 1
    arr = np.random.randn(2, 3, 2).astype(np.float32)
    obj = np.array([1], dtype=np.int32)
    axis = 1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with negative axis
    arr = np.random.randn(4, 4).astype(np.float32)
    obj = np.array([3], dtype=np.int32)
    axis = -1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, axis 2, unique indices
    arr = np.random.randn(2, 2, 3, 2).astype(np.float32)
    obj = np.array([0, 2], dtype=np.int32)
    axis = 2
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
