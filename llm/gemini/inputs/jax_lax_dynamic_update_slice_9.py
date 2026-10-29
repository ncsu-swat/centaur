
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive start index
    operand = np.zeros((8,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    allow_negative_indices = (True,)
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

    # Input 2: 1D int32 array, negative start index
    operand = np.arange(10, dtype=np.int32)
    update = np.array([-1, -2, -3], dtype=np.int32)
    start_indices = np.array([-4], dtype=np.int32)
    allow_negative_indices = (True,)
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

    # Input 3: 2D float32 array, non-negative indices
    operand = np.zeros((6, 6), dtype=np.float32)
    update = np.ones((2, 3), dtype=np.float32)
    start_indices = np.array([1, 2], dtype=np.int32)
    allow_negative_indices = (False, False)
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

    # Input 4: 2D int64 array, mixed negative and positive start indices
    operand = np.arange(20, dtype=np.int64).reshape((4, 5))
    update = np.full((2, 2), 99, dtype=np.int64)
    start_indices = np.array([-2, 1], dtype=np.int64)
    allow_negative_indices = (True, True)
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

    # Input 5: 3D float64 array, mixed allow_negative_indices
    operand = np.random.randn(5, 5, 5).astype(np.float64)
    update = np.random.randn(2, 3, 1).astype(np.float64)
    start_indices = np.array([1, 0, -2], dtype=np.int32)
    allow_negative_indices = (True, False, True)
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

    # Input 6: 3D bool array
    operand = np.zeros((3, 4, 4), dtype=bool)
    update = np.ones((1, 2, 2), dtype=bool)
    start_indices = np.array([0, 1, 1], dtype=np.int32)
    allow_negative_indices = (False, False, False)
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

    # Input 7: 4D float32 array
    operand = np.zeros((2, 4, 4, 3), dtype=np.float32)
    update = np.ones((1, 2, 2, 1), dtype=np.float32)
    start_indices = np.array([0, 1, 2, 0], dtype=np.int32)
    allow_negative_indices = (True, True, True, True)
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

    # Input 8: 2D complex64 array
    operand = np.zeros((4, 4), dtype=np.complex64)
    update = np.ones((2, 2), dtype=np.complex64) * (1 + 2j)
    start_indices = np.array([-1, 1], dtype=np.int32)
    allow_negative_indices = (True, False)
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

    # Input 9: 1D uint8 array, start index out of bounds to trigger clamping
    operand = np.zeros((10,), dtype=np.uint8)
    update = np.full((4,), 255, dtype=np.uint8)
    start_indices = np.array([8], dtype=np.int32)
    allow_negative_indices = (False,)
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

    # Input 10: 2D float16 array, exact full size update
    operand = np.ones((3, 3), dtype=np.float16)
    update = np.zeros((3, 3), dtype=np.float16)
    start_indices = np.array([0, 0], dtype=np.int32)
    allow_negative_indices = (False, True)
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

    # Input 11: 3D int16 array with all negative start indices
    operand = np.arange(27, dtype=np.int16).reshape((3, 3, 3))
    update = np.zeros((1, 1, 1), dtype=np.int16)
    start_indices = np.array([-1, -2, -3], dtype=np.int32)
    allow_negative_indices = (True, True, True)
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
