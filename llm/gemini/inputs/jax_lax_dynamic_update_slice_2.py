
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_update_slice_inputs():
    list_of_inputs = []

    # 1. 1D float32 array
    operand = np.zeros(6, dtype=np.float32)
    update = np.ones(3, dtype=np.float32)
    start_indices = (np.int32(2),)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 2. 2D float32 array with negative start index
    operand = np.zeros((4, 5), dtype=np.float32)
    update = np.ones((2, 2), dtype=np.float32)
    start_indices = (np.int32(-2), np.int32(1))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 3. 2D int32 array, allow_negative_indices=False
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    update = -np.ones((2, 2), dtype=np.int32)
    start_indices = (np.int64(1), np.int64(2))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": False,
            }
        )
    )

    # 4. 3D float64 array
    operand = np.zeros((3, 4, 5), dtype=np.float64)
    update = np.ones((1, 2, 3), dtype=np.float64)
    start_indices = (np.int32(0), np.int32(1), np.int32(2))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 5. 1D int64 array with out-of-bounds start index (XLA clamps it)
    operand = np.arange(10, dtype=np.int64)
    update = np.array([99, 100], dtype=np.int64)
    start_indices = (np.int32(15),)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 6. 4D bool array
    operand = np.zeros((2, 3, 2, 4), dtype=bool)
    update = np.ones((1, 1, 1, 2), dtype=bool)
    start_indices = (np.int32(1), np.int32(2), np.int32(0), np.int32(1))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 7. 2D complex64 array
    operand = np.zeros((4, 4), dtype=np.complex64)
    update = np.ones((2, 2), dtype=np.complex64) * (1.0 + 2.0j)
    start_indices = (np.int32(0), np.int32(0))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": False,
            }
        )
    )

    # 8. Full shape update (update has identical shape to operand)
    operand = np.ones((3, 3), dtype=np.float32)
    update = np.zeros((3, 3), dtype=np.float32)
    start_indices = (np.int32(0), np.int32(0))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 9. 1D single-element update
    operand = np.arange(8, dtype=np.float32)
    update = np.array([42.0], dtype=np.float32)
    start_indices = (np.int32(-1),)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": True,
            }
        )
    )

    # 10. 3D uint8 array
    operand = np.full((5, 5, 5), 10, dtype=np.uint8)
    update = np.full((2, 3, 1), 255, dtype=np.uint8)
    start_indices = (np.int32(2), np.int32(1), np.int32(3))
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "update": update,
                "start_indices": start_indices,
                "allow_negative_indices": False,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.dynamic_update_slice_2"] = (
    dynamic_update_slice_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_2'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_2'], lib="jax", suffix=2)
