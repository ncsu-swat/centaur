
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive start index
    operand = np.zeros((10,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    allow_negative_indices = [True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 2: 2D int32 array, standard update
    operand = np.arange(20, dtype=np.int32).reshape(4, 5)
    update = -np.ones((2, 3), dtype=np.int32)
    start_indices = np.array([1, 2], dtype=np.int32)
    allow_negative_indices = [True, True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 3: 2D float64 array, negative start indices
    operand = np.zeros((6, 6), dtype=np.float64)
    update = np.full((2, 2), 5.0, dtype=np.float64)
    start_indices = np.array([-2, -3], dtype=np.int64)
    allow_negative_indices = [True, True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 4: 3D float32 array, allow_negative_indices mixed False/True
    operand = np.random.randn(4, 5, 6).astype(np.float32)
    update = np.ones((2, 2, 3), dtype=np.float32)
    start_indices = np.array([0, 1, 2], dtype=np.int32)
    allow_negative_indices = [False, True, False]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 5: 1D bool array, full update
    operand = np.zeros((5,), dtype=np.bool_)
    update = np.ones((5,), dtype=np.bool_)
    start_indices = np.array([0], dtype=np.int32)
    allow_negative_indices = [False]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 6: 4D float32 array, unit update slice size
    operand = np.zeros((3, 4, 5, 2), dtype=np.float32)
    update = np.ones((1, 1, 1, 1), dtype=np.float32)
    start_indices = np.array([2, 1, 3, 0], dtype=np.int32)
    allow_negative_indices = [True, True, True, True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 7: 2D complex64 array
    operand = np.zeros((5, 4), dtype=np.complex64)
    update = np.ones((2, 2), dtype=np.complex64) + 2j
    start_indices = np.array([1, 0], dtype=np.int32)
    allow_negative_indices = [False, False]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 8: 1D int64 array, out-of-bounds start index (clamped by dynamic_update_slice)
    operand = np.arange(8, dtype=np.int64)
    update = np.array([100, 200, 300], dtype=np.int64)
    start_indices = np.array([15], dtype=np.int64)
    allow_negative_indices = [True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 9: 3D uint8 array
    operand = np.zeros((8, 8, 3), dtype=np.uint8)
    update = np.full((4, 4, 3), 255, dtype=np.uint8)
    start_indices = np.array([2, 2, 0], dtype=np.int32)
    allow_negative_indices = [True, True, True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 10: 0D scalar arrays
    operand = np.array(42.0, dtype=np.float32)
    update = np.array(7.0, dtype=np.float32)
    start_indices = np.array([], dtype=np.int32)
    allow_negative_indices = []
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 11: 5D float32 array
    operand = np.zeros((2, 3, 4, 5, 2), dtype=np.float32)
    update = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
    start_indices = np.array([0, 1, 1, 2, 1], dtype=np.int32)
    allow_negative_indices = [True, False, True, False, True]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

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
