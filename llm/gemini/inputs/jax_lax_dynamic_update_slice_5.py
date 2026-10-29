
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
    start_indices = (np.int32(2),)
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

    # Input 2: 2D int32 array, normal start indices
    operand = np.arange(20, dtype=np.int32).reshape((4, 5))
    update = np.ones((2, 2), dtype=np.int32)
    start_indices = (np.int32(1), np.int32(2))
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

    # Input 3: 3D float64 array, mixed allow_negative_indices
    operand = np.zeros((3, 4, 5), dtype=np.float64)
    update = np.ones((1, 2, 3), dtype=np.float64)
    start_indices = (np.int32(0), np.int32(1), np.int32(2))
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

    # Input 4: 1D negative index with allow_negative_indices=True
    operand = np.arange(8, dtype=np.float32)
    update = np.array([10.0, 20.0], dtype=np.float32)
    start_indices = (np.int32(-3),)
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

    # Input 5: 2D with negative index on one dimension
    operand = np.zeros((6, 6), dtype=np.int64)
    update = np.full((3, 2), 5, dtype=np.int64)
    start_indices = (np.int32(-4), np.int32(1))
    allow_negative_indices = [True, False]
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

    # Input 6: 4D uint8 array
    operand = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    update = np.ones((1, 2, 2, 3), dtype=np.uint8)
    start_indices = (np.int32(0), np.int32(1), np.int32(1), np.int32(0))
    allow_negative_indices = [False, False, False, False]
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

    # Input 7: 1D full replacement (update shape equals operand shape)
    operand = np.ones((5,), dtype=np.int16)
    update = np.zeros((5,), dtype=np.int16)
    start_indices = (np.int32(0),)
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

    # Input 8: 2D boolean array
    operand = np.zeros((4, 4), dtype=bool)
    update = np.ones((2, 3), dtype=bool)
    start_indices = (np.int32(1), np.int32(0))
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

    # Input 9: 2D complex64 array with out-of-bounds start index (clamped by dynamic_update_slice)
    operand = np.zeros((5, 5), dtype=np.complex64)
    update = np.ones((2, 2), dtype=np.complex64)
    start_indices = (np.int32(10), np.int32(4))
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

    # Input 10: 5D float32 array
    operand = np.zeros((2, 2, 3, 3, 4), dtype=np.float32)
    update = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    start_indices = (
        np.int32(1),
        np.int32(0),
        np.int32(1),
        np.int32(0),
        np.int32(2),
    )
    allow_negative_indices = [True, True, True, True, True]
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


generated_inputs["jax.lax.dynamic_update_slice_5"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_5'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_5'], lib="jax", suffix=5)
