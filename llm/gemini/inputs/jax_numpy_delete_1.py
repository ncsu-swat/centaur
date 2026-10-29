
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, delete first element along axis 0
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = 0
    axis = 0
    assume_unique_indices = False
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array, negative axis and positive index
    arr = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    obj = 2
    axis = -1
    assume_unique_indices = True
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, delete row along axis 0
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    obj = 1
    axis = 0
    assume_unique_indices = False
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, delete column using negative index along axis 1
    arr = np.random.randn(4, 5).astype(np.float64)
    obj = -1
    axis = 1
    assume_unique_indices = True
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative axis index
    arr = np.ones((6, 3), dtype=np.uint8)
    obj = 0
    axis = -2
    assume_unique_indices = False
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float array, axis 0
    arr = np.random.randn(4, 3, 2).astype(np.float64)
    obj = 2
    axis = 0
    assume_unique_indices = True
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D integer array, negative index along axis 1
    arr = np.arange(30, dtype=np.int32).reshape(2, 5, 3)
    obj = -2
    axis = 1
    assume_unique_indices = False
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex array, axis 2
    arr = (np.random.randn(2, 2, 4) + 1j * np.random.randn(2, 2, 4)).astype(
        np.complex64
    )
    obj = 1
    axis = 2
    assume_unique_indices = True
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D boolean array, axis 3
    arr = np.random.choice([True, False], size=(2, 3, 4, 5))
    obj = 0
    axis = 3
    assume_unique_indices = False
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float array, negative index and negative axis
    arr = np.random.randn(1, 2, 3, 4).astype(np.float32)
    obj = -3
    axis = -1
    assume_unique_indices = True
    input_dict = {
        "arr": arr,
        "obj": obj,
        "axis": axis,
        "assume_unique_indices": assume_unique_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.delete_1"] = delete_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.delete_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.delete_1'.")


check_valid('jax.numpy.delete', generated_inputs['jax.numpy.delete_1'], lib="jax", suffix=1)
