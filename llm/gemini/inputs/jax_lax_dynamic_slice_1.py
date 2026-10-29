
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax


def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, normal slice
    operand = np.arange(10, dtype=np.float32)
    start_indices = (2,)
    slice_sizes = (4,)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 2: 2D int32 array, standard dynamic slice
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    start_indices = (1, 1)
    slice_sizes = (2, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 3: 2D float64 array with negative indices
    operand = np.random.randn(8, 8).astype(np.float64)
    start_indices = (-4, -2)
    slice_sizes = (3, 2)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 4: 2D int64 array with allow_negative_indices=False
    operand = np.arange(20, dtype=np.int64).reshape(4, 5)
    start_indices = (0, 2)
    slice_sizes = (3, 2)
    allow_negative_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 5: 3D float32 array
    operand = np.random.randn(4, 5, 6).astype(np.float32)
    start_indices = (1, 2, 0)
    slice_sizes = (2, 2, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 6: 4D float16 array
    operand = np.random.randn(2, 4, 4, 3).astype(np.float16)
    start_indices = (0, 1, 1, 0)
    slice_sizes = (1, 2, 2, 2)
    allow_negative_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 7: 5D bool array
    operand = np.ones((2, 3, 2, 3, 2), dtype=np.bool_)
    start_indices = (0, 1, 0, 1, 0)
    slice_sizes = (1, 1, 2, 1, 1)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 8: Complex64 1D array with zero-length slice
    operand = (np.arange(8) + 1j * np.arange(8)).astype(np.complex64)
    start_indices = (3,)
    slice_sizes = (0,)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 9: uint8 full-size slice
    operand = np.arange(16, dtype=np.uint8).reshape(4, 4)
    start_indices = (0, 0)
    slice_sizes = (4, 4)
    allow_negative_indices = False
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 10: 2D array overrun clamped slice
    operand = np.arange(25, dtype=np.int16).reshape(5, 5)
    start_indices = (4, 3)
    slice_sizes = (2, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    # Input 11: 3D int8 array with negative indices
    operand = np.ones((6, 6, 6), dtype=np.int8)
    start_indices = (-3, -4, -2)
    slice_sizes = (2, 3, 1)
    allow_negative_indices = True
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "slice_sizes": slice_sizes,
                "allow_negative_indices": allow_negative_indices,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.dynamic_slice_1"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_1'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_1'], lib="jax", suffix=1)
