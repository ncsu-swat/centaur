
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def delete_inputs():
    list_of_inputs = []

    # Input 1: 1D array, delete index 2 along axis 0
    arr = np.array([4, 5, 6, 7, 8], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "obj": int(2),
        "axis": int(0),
        "assume_unique_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, delete index 1 along axis 0, with unique indices assumed
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "obj": int(1),
        "axis": int(0),
        "assume_unique_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, negative index along axis 1
    arr = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "arr": arr,
        "obj": int(-1),
        "axis": int(1),
        "assume_unique_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array, delete index 0 along axis 2
    arr = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "obj": int(0),
        "axis": int(2),
        "assume_unique_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array, negative index and negative axis
    arr = np.array([10, 20, 30, 40], dtype=np.float64)
    input_dict = {
        "arr": arr,
        "obj": int(-2),
        "axis": int(-1),
        "assume_unique_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float32 array, delete index 2 along axis 0
    arr = np.random.randn(3, 3).astype(np.float32)
    input_dict = {
        "arr": arr,
        "obj": int(2),
        "axis": int(0),
        "assume_unique_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, negative axis
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "arr": arr,
        "obj": int(1),
        "axis": int(-2),
        "assume_unique_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D int32 array, delete index 0 along axis 3
    arr = np.random.randint(0, 5, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "obj": int(0),
        "axis": int(3),
        "assume_unique_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 array, index 3, assume unique indices
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    input_dict = {
        "arr": arr,
        "obj": int(3),
        "axis": int(0),
        "assume_unique_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 2D array, deleting last index along axis 1
    arr = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "arr": arr,
        "obj": int(9),
        "axis": int(1),
        "assume_unique_indices": True
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
