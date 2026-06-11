
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, positive start index, negative indices allowed
    operand = np.arange(10, dtype=np.float32)
    update = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    allow_negative_indices = [True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 2: 1D array, int32, negative start index
    operand = np.arange(8, dtype=np.int32)
    update = np.array([-10, -20], dtype=np.int32)
    start_indices = np.array([-3], dtype=np.int32)
    allow_negative_indices = [True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 3: 2D array, float64, mixed allow_negative_indices
    operand = np.ones((5, 5), dtype=np.float64)
    update = np.zeros((2, 2), dtype=np.float64)
    start_indices = np.array([1, 2], dtype=np.int64)
    allow_negative_indices = [False, False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 4: 2D array, int64, out-of-bound start index (to test automatic adjustment/clamping)
    operand = np.zeros((4, 4), dtype=np.int64)
    update = np.ones((2, 3), dtype=np.int64)
    start_indices = np.array([3, 2], dtype=np.int32)
    allow_negative_indices = [True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 5: 3D array, float32, all positive indices
    operand = np.random.randn(4, 4, 4).astype(np.float32)
    update = np.ones((2, 2, 2), dtype=np.float32)
    start_indices = np.array([0, 1, 2], dtype=np.int32)
    allow_negative_indices = [True, True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 6: 1D array, boolean dtype
    operand = np.array([True, False, True, False, True, False], dtype=bool)
    update = np.array([True, True, True], dtype=bool)
    start_indices = np.array([3], dtype=np.int32)
    allow_negative_indices = [False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 7: 4D array, float32, diverse shapes
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    update = np.zeros((1, 2, 2, 2), dtype=np.float32)
    start_indices = np.array([1, 0, 1, 2], dtype=np.int32)
    allow_negative_indices = [True, False, True, False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 8: 2D array, complex64
    operand = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    update = np.array([[1.0 + 1j, 2.0 - 2j]], dtype=np.complex64)
    start_indices = np.array([2, 1], dtype=np.int32)
    allow_negative_indices = [True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 9: 1D array, negative index but allow_negative_indices is False (clamps to 0)
    operand = np.arange(10, dtype=np.float32)
    update = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    start_indices = np.array([-2], dtype=np.int32)
    allow_negative_indices = [False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Input 10: 5D array, float32, update size 1 along all dimensions
    operand = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)
    update = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    start_indices = np.array([1, 1, 0, 1, 0], dtype=np.int32)
    allow_negative_indices = [True, True, True, True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_6"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_6'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_6'], lib="jax", suffix=6)
