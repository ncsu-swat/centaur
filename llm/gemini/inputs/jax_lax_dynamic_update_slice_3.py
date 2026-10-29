
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive index
    operand = np.zeros((10,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, allow_negative_indices=False
    operand = np.arange(8, dtype=np.int32)
    update = np.array([99, 100], dtype=np.int32)
    start_indices = np.array([4], dtype=np.int32)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, negative index
    operand = np.zeros((5, 5), dtype=np.float32)
    update = np.full((2, 2), 3.5, dtype=np.float32)
    start_indices = np.array([-2, -2], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array
    operand = np.ones((6, 4), dtype=np.int64)
    update = np.zeros((3, 2), dtype=np.int64)
    start_indices = np.array([1, 1], dtype=np.int64)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array
    operand = np.random.randn(4, 6, 8).astype(np.float64)
    update = np.random.randn(2, 3, 4).astype(np.float64)
    start_indices = np.array([1, 2, 3], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, index at boundary
    operand = np.zeros((3, 4, 5), dtype=np.float32)
    update = np.ones((1, 2, 2), dtype=np.float32)
    start_indices = np.array([0, 0, 0], dtype=np.int32)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    operand = np.zeros((2, 3, 4, 5), dtype=np.float32)
    update = np.ones((1, 2, 2, 3), dtype=np.float32)
    start_indices = np.array([1, 0, 1, 2], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array
    operand = np.zeros((4, 4), dtype=bool)
    update = np.ones((2, 3), dtype=bool)
    start_indices = np.array([2, 1], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex64 array
    operand = np.zeros((8,), dtype=np.complex64)
    update = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    start_indices = np.array([3], dtype=np.int32)
    allow_negative_indices = False
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D int32 array with mixed negative and positive indices
    operand = np.zeros((6, 7, 8), dtype=np.int32)
    update = np.full((3, 2, 4), 7, dtype=np.int32)
    start_indices = np.array([-4, 2, -5], dtype=np.int32)
    allow_negative_indices = True
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_3"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_3'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_3'], lib="jax", suffix=3)
