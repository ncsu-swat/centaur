
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, basic positive index
    operand = np.zeros((6,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = [np.int32(2)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32, negative start index with allow_negative_indices=True
    operand = np.arange(10, dtype=np.int32)
    update = np.array([99, 100], dtype=np.int32)
    start_indices = [np.int32(-3)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32, allow_negative_indices=False
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = [np.int32(1), np.int32(2)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex64, negative indices
    operand = np.zeros((5, 5), dtype=np.complex64)
    update = np.ones((2, 3), dtype=np.complex64)
    start_indices = [np.int32(-2), np.int32(-3)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int64, interior slice update
    operand = np.zeros((4, 6, 8), dtype=np.int64)
    update = np.ones((2, 3, 4), dtype=np.int64)
    start_indices = [np.int64(1), np.int64(2), np.int64(3)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64, allow_negative_indices=False
    operand = np.arange(27, dtype=np.float64).reshape((3, 3, 3))
    update = np.zeros((1, 2, 2), dtype=np.float64)
    start_indices = [np.int32(0), np.int32(1), np.int32(1)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D bool array
    operand = np.zeros((2, 3, 4, 5), dtype=np.bool_)
    update = np.ones((1, 2, 2, 2), dtype=np.bool_)
    start_indices = [np.int32(0), np.int32(1), np.int32(2), np.int32(1)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Full update (update shape equals operand shape)
    operand = np.zeros((5,), dtype=np.float32)
    update = np.arange(5, dtype=np.float32)
    start_indices = [np.int32(0)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Out-of-bounds start index (tests clamping behavior)
    operand = np.ones((6, 6), dtype=np.int32)
    update = np.zeros((2, 2), dtype=np.int32)
    start_indices = [np.int32(10), np.int32(20)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    operand = np.zeros((2, 2, 3, 4, 5), dtype=np.float32)
    update = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    start_indices = [np.int32(1), np.int32(0), np.int32(1), np.int32(2), np.int32(3)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D uint8 array with mixed indices
    operand = np.full((8, 8), 255, dtype=np.uint8)
    update = np.zeros((3, 3), dtype=np.uint8)
    start_indices = [np.int32(-5), np.int32(2)]
    input_dict = {
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_1"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_1'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_1'], lib="jax", suffix=1)
