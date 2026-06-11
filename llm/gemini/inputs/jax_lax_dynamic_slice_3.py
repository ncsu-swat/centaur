
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive index
    operand = np.arange(10).astype(np.float32)
    start_indices = np.array([2], dtype=np.int32)
    slice_sizes = (4,)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, positive indices
    operand = np.random.randn(5, 5).astype(np.float32)
    start_indices = np.array([1, 1], dtype=np.int32)
    slice_sizes = (3, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array, int64 indices, no negative indices allowed
    operand = np.arange(24).reshape(2, 3, 4).astype(np.int32)
    start_indices = np.array([0, 1, 2], dtype=np.int64)
    slice_sizes = (1, 2, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, negative start index
    operand = np.random.randn(10, 10, 10).astype(np.float64)
    start_indices = np.array([2, -1, 3], dtype=np.int32)
    slice_sizes = (4, 2, 5)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D boolean array
    operand = np.random.choice([True, False], size=(6, 6))
    start_indices = np.array([0, 0], dtype=np.int32)
    slice_sizes = (6, 6)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, larger bounds
    operand = np.arange(100).reshape(2, 5, 10).astype(np.float32)
    start_indices = np.array([1, 2, 5], dtype=np.int32)
    slice_sizes = (1, 2, 3)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D int64 array, negative indices
    operand = np.arange(16).reshape(4, 4).astype(np.int64)
    start_indices = np.array([-2, -2], dtype=np.int32)
    slice_sizes = (2, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array
    operand = np.random.randn(3, 3, 3, 3).astype(np.float32)
    start_indices = np.array([1, 1, 1, 1], dtype=np.int32)
    slice_sizes = (2, 2, 2, 2)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int32 array, negative index
    operand = np.arange(8).astype(np.int32)
    start_indices = np.array([-5], dtype=np.int32)
    slice_sizes = (3,)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 array, no negative indices allowed
    operand = np.random.randn(8, 12).astype(np.float32)
    start_indices = np.array([4, 6], dtype=np.int32)
    slice_sizes = (2, 4)
    input_dict = {
        "operand": operand,
        "start_indices": start_indices,
        "slice_sizes": slice_sizes,
        "allow_negative_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_slice_3"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_3'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_3'], lib="jax", suffix=3)
