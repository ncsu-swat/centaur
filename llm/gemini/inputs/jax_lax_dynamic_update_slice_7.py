
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, positive index
    operand = np.zeros(10, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = [2]
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

    # Input 2: 1D int32, negative index
    operand = np.arange(8, dtype=np.int32)
    update = np.array([-1, -2], dtype=np.int32)
    start_indices = [-3]
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

    # Input 3: 2D float64, standard update
    operand = np.zeros((6, 6), dtype=np.float64)
    update = np.ones((2, 3), dtype=np.float64)
    start_indices = [1, 2]
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

    # Input 4: 2D boolean array, negative index in one dimension
    operand = np.zeros((4, 5), dtype=bool)
    update = np.ones((2, 2), dtype=bool)
    start_indices = [-2, 1]
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

    # Input 5: 3D int64, zero indices
    operand = np.ones((4, 4, 4), dtype=np.int64)
    update = np.full((2, 1, 3), 9, dtype=np.int64)
    start_indices = [0, 0, 0]
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

    # Input 6: 3D float32, all negative start indices
    operand = np.random.randn(5, 5, 5).astype(np.float32)
    update = np.random.randn(2, 2, 2).astype(np.float32)
    start_indices = [-3, -4, -2]
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

    # Input 7: 4D complex64, small update slice
    operand = (
        np.random.randn(3, 4, 3, 2) + 1j * np.random.randn(3, 4, 3, 2)
    ).astype(np.complex64)
    update = (
        np.random.randn(1, 2, 1, 1) + 1j * np.random.randn(1, 2, 1, 1)
    ).astype(np.complex64)
    start_indices = [1, 1, 1, 0]
    allow_negative_indices = (True, False, True, False)
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

    # Input 8: 2D uint8, update slice equal to operand shape
    operand = np.zeros((3, 3), dtype=np.uint8)
    update = np.ones((3, 3), dtype=np.uint8) * 255
    start_indices = [0, 0]
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

    # Input 9: 1D float16, single element update
    operand = np.linspace(0, 1, 10, dtype=np.float16)
    update = np.array([9.9], dtype=np.float16)
    start_indices = [5]
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

    # Input 10: 5D int16, diverse dimensions
    operand = np.zeros((2, 3, 4, 3, 2), dtype=np.int16)
    update = np.ones((1, 1, 2, 1, 1), dtype=np.int16)
    start_indices = [1, 2, 1, 0, 1]
    allow_negative_indices = (True, True, True, True, True)
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

    # Input 11: 2D int32, clamping behavior with large start_indices
    operand = np.zeros((5, 5), dtype=np.int32)
    update = np.ones((2, 2), dtype=np.int32)
    start_indices = [10, 10]
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

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_7"] = (
    dynamic_update_slice_inputs()
)

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
