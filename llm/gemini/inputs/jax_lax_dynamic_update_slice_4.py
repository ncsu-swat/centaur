
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    operand = np.zeros((10,), dtype=np.float32)
    update = np.ones((3,), dtype=np.float32)
    start_indices = [2]
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

    # Input 2: 1D int32 array with negative start index
    operand = np.arange(8, dtype=np.int32)
    update = np.array([-1, -2], dtype=np.int32)
    start_indices = [-2]
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

    # Input 3: 2D float64 array
    operand = np.zeros((6, 6), dtype=np.float64)
    update = np.ones((2, 3), dtype=np.float64)
    start_indices = [1, 2]
    allow_negative_indices = [False, True]
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

    # Input 4: 2D int64 array at top-left corner
    operand = np.zeros((4, 5), dtype=np.int64)
    update = np.full((2, 2), 42, dtype=np.int64)
    start_indices = [0, 0]
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

    # Input 5: 3D float32 array
    operand = np.zeros((4, 4, 4), dtype=np.float32)
    update = np.ones((2, 2, 2), dtype=np.float32)
    start_indices = [1, 1, 1]
    allow_negative_indices = [False, False, False]
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

    # Input 6: 3D complex64 array with negative indices
    operand = np.zeros((3, 5, 4), dtype=np.complex64)
    update = np.ones((1, 2, 2), dtype=np.complex64)
    start_indices = [1, -2, 0]
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

    # Input 7: 4D uint8 array
    operand = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    update = np.ones((1, 2, 2, 3), dtype=np.uint8)
    start_indices = [0, 1, 0, 2]
    allow_negative_indices = [True, False, True, False]
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

    # Input 8: 1D bool array
    operand = np.zeros(6, dtype=bool)
    update = np.ones(3, dtype=bool)
    start_indices = [0]
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

    # Input 9: 2D float16 array with mixed negative indices
    operand = np.zeros((5, 5), dtype=np.float16)
    update = np.ones((2, 2), dtype=np.float16)
    start_indices = [-2, -3]
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

    # Input 10: 3D int16 array with scalar array start indices
    operand = np.zeros((3, 3, 3), dtype=np.int16)
    update = np.ones((1, 1, 1), dtype=np.int16)
    start_indices = [np.int32(2), np.int32(0), np.int32(1)]
    allow_negative_indices = [True, False, True]
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

    # Input 11: 2D array full slice replacement
    operand = np.zeros((3, 3), dtype=np.float32)
    update = np.ones((3, 3), dtype=np.float32)
    start_indices = [0, 0]
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

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_4"] = (
    dynamic_update_slice_inputs()
)

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
