
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, within bounds
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = [np.int32(2)]
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 2: 1D array, float32, negative start index
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = [np.int32(-3)]
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 3: 2D array, float32, normal update
    operand = np.zeros((4, 4), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = [np.int32(1), np.int32(2)]
    allow_negative_indices = (True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 4: 2D array, int32, negative index allowed
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    update = np.ones((2, 2), dtype=np.int32)
    start_indices = [np.int32(-2), np.int32(1)]
    allow_negative_indices = (True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 5: 3D array, float64, mixed allow_negative_indices
    operand = np.zeros((3, 3, 3), dtype=np.float64)
    update = np.ones((2, 1, 2), dtype=np.float64)
    start_indices = [np.int32(1), np.int32(0), np.int32(1)]
    allow_negative_indices = (True, False, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 6: 1D array, float32, out of bounds with clamping
    operand = np.zeros(5, dtype=np.float32)
    update = np.ones(2, dtype=np.float32)
    start_indices = [np.int32(4)]
    allow_negative_indices = (False,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 7: 4D array, bool
    operand = np.zeros((2, 2, 2, 2), dtype=bool)
    update = np.ones((1, 1, 1, 1), dtype=bool)
    start_indices = [np.int32(0), np.int32(1), np.int32(0), np.int32(1)]
    allow_negative_indices = (True, True, True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 8: 2D array, large size, float16
    operand = np.zeros((100, 100), dtype=np.float16)
    update = np.ones((10, 10), dtype=np.float16)
    start_indices = [np.int32(50), np.int32(50)]
    allow_negative_indices = (True, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 9: 3D array, negative indexing with False for one dimension
    operand = np.zeros((5, 5, 5), dtype=np.float32)
    update = np.ones((2, 2, 2), dtype=np.float32)
    start_indices = [np.int32(-2), np.int32(1), np.int32(-1)]
    allow_negative_indices = (True, False, True)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    # Input 10: 1D array, int64, index at end
    operand = np.arange(10, dtype=np.int64)
    update = np.array([100, 200], dtype=np.int64)
    start_indices = [np.int32(8)]
    allow_negative_indices = (True,)
    list_of_inputs.append({
        'operand': operand,
        'update': update,
        'start_indices': start_indices,
        'allow_negative_indices': allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_7"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_7'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_7'], lib="jax", suffix=7)
