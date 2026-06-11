
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 2: 1D float32 array with negative start index
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = np.array([-3], dtype=np.int32)
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 3: 2D float32 array
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = np.array([1, 2], dtype=np.int32)
    allow_negative_indices = (True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 4: 2D float64 array, allow_negative_indices is False
    operand = np.zeros((5, 5), dtype=np.float64)
    update = np.ones((3, 2), dtype=np.float64)
    start_indices = np.array([2, 1], dtype=np.int32)
    allow_negative_indices = (False, False)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 5: 3D float32 array
    operand = np.zeros((3, 3, 3), dtype=np.float32)
    update = np.ones((1, 2, 2), dtype=np.float32)
    start_indices = np.array([1, 0, 1], dtype=np.int32)
    allow_negative_indices = (True, True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 6: 1D int32 array
    operand = np.arange(10, dtype=np.int32)
    update = np.array([99, 99], dtype=np.int32)
    start_indices = np.array([4], dtype=np.int32)
    allow_negative_indices = (False,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 7: 4D float32 array
    operand = np.zeros((2, 3, 4, 5), dtype=np.float32)
    update = np.ones((1, 1, 2, 2), dtype=np.float32)
    start_indices = np.array([0, 1, 2, 3], dtype=np.int32)
    allow_negative_indices = (True, True, True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 8: 1D array with clamping start index
    operand = np.zeros(5, dtype=np.float32)
    update = np.ones(2, dtype=np.float32)
    start_indices = np.array([10], dtype=np.int32)
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 9: 2D int64 array with negative indices
    operand = np.zeros((6, 6), dtype=np.int64)
    update = np.ones((2, 3), dtype=np.int64)
    start_indices = np.array([-2, -4], dtype=np.int32)
    allow_negative_indices = (True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 10: 1D bool array
    operand = np.zeros(5, dtype=bool)
    update = np.ones(2, dtype=bool)
    start_indices = np.array([1], dtype=np.int32)
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 11: 3D int16 array with mixed allow_negative_indices values
    operand = np.zeros((4, 4, 4), dtype=np.int16)
    update = np.ones((2, 2, 2), dtype=np.int16)
    start_indices = np.array([1, 1, 1], dtype=np.int32)
    allow_negative_indices = (True, False, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_9"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_9'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_9'], lib="jax", suffix=9)
