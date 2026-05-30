
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def delete_inputs():
    list_of_inputs = []

    # Input 1
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    obj = np.array([1, 3], dtype=np.int32)
    axis = 0
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    arr = np.random.randn(4, 3).astype(np.float32)
    obj = np.array([0, 2], dtype=np.int32)
    axis = 0
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    arr = np.arange(12).reshape(4, 3).astype(np.int32)
    obj = np.array([1], dtype=np.int32)
    axis = 1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    obj = np.array([0, 2, 3], dtype=np.int32)
    axis = 2
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    arr = np.random.randint(0, 10, size=(3, 3, 3)).astype(np.int64)
    obj = np.array([1, 2], dtype=np.int32)
    axis = -1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    arr = np.random.randn(5, 2, 2, 2).astype(np.float32)
    obj = np.array([4], dtype=np.int32)
    axis = 0
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    arr = np.random.randn(5, 5).astype(np.float32)
    obj = np.array([0, 1, 4], dtype=np.int32)
    axis = -2
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    arr = np.arange(10).astype(np.float32)
    obj = np.array([5], dtype=np.int32)
    axis = 0
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    arr = np.random.randn(3, 6).astype(np.float32)
    obj = np.array([0, 2, 4, 5], dtype=np.int32)
    axis = 1
    assume_unique_indices = True
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    arr = np.random.randn(2, 4, 2).astype(np.float32)
    obj = np.array([1, 3], dtype=np.int32)
    axis = 1
    assume_unique_indices = False
    input_dict = {"arr": arr, "obj": obj, "axis": axis, "assume_unique_indices": assume_unique_indices}
    list_of_inputs.append(copy.deepcopy(input_dict))

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
