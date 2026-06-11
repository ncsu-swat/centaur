
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Case 1: 1D, float32, simple update
    operand = np.zeros((10,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = [2]
    allow_negative_indices = [True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 2: 2D, float32, square shapes
    operand = np.random.randn(5, 5).astype(np.float32)
    update = np.random.randn(2, 2).astype(np.float32)
    start_indices = [1, 2]
    allow_negative_indices = [True, False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 3: 3D, int32
    operand = np.arange(64, dtype=np.int32).reshape((4, 4, 4))
    update = np.ones((2, 1, 2), dtype=np.int32)
    start_indices = [0, 1, 2]
    allow_negative_indices = [True, True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 4: 1D, float64, negative start index
    operand = np.arange(8, dtype=np.float64)
    update = np.ones((4,), dtype=np.float64) * 9.0
    start_indices = [-3]
    allow_negative_indices = [True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 5: 4D, float32
    operand = np.random.randn(2, 3, 4, 5).astype(np.float32)
    update = np.random.randn(1, 2, 2, 1).astype(np.float32)
    start_indices = [1, 1, 2, 3]
    allow_negative_indices = [False, False, False, False]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 6: 2D, complex64
    operand = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    update = (np.random.randn(1, 1) + 1j * np.random.randn(1, 1)).astype(np.complex64)
    start_indices = [2, 2]
    allow_negative_indices = [True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 7: 1D, int64, large array
    operand = np.arange(100, dtype=np.int64)
    update = np.zeros((10,), dtype=np.int64)
    start_indices = [50]
    allow_negative_indices = [True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 8: 3D, float16, mixed sign start indices
    operand = np.random.randn(3, 3, 3).astype(np.float16)
    update = np.random.randn(2, 2, 2).astype(np.float16)
    start_indices = [-1, -2, 0]
    allow_negative_indices = [True, True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 9: 2D, uint8
    operand = np.ones((10, 10), dtype=np.uint8) * 5
    update = np.zeros((3, 3), dtype=np.uint8)
    start_indices = [5, 5]
    allow_negative_indices = [False, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    # Case 10: 5D, float32, single element update
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    update = np.random.randn(1, 1, 1, 1, 1).astype(np.float32)
    start_indices = [0, 1, 0, 1, 0]
    allow_negative_indices = [True, True, True, True, True]
    list_of_inputs.append({
        "operand": operand,
        "update": update,
        "start_indices": start_indices,
        "allow_negative_indices": allow_negative_indices
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_4"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_4'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_4'], lib="jax", suffix=4)
