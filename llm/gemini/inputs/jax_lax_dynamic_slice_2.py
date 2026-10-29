
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D array, basic slice
    operand = np.arange(10, dtype=np.int32)
    start_indices = [np.int32(2)]
    slice_sizes = [4]
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

    # Input 2: 2D array, float32, positive indices
    operand = np.random.randn(5, 6).astype(np.float32)
    start_indices = [np.int32(1), np.int32(2)]
    slice_sizes = [2, 3]
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

    # Input 3: 2D array, negative start indices with allow_negative_indices=True
    operand = np.arange(20, dtype=np.float64).reshape(4, 5)
    start_indices = [np.int32(-2), np.int32(-3)]
    slice_sizes = [2, 2]
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

    # Input 4: 3D array, int64
    operand = np.arange(60, dtype=np.int64).reshape(3, 4, 5)
    start_indices = [np.int32(0), np.int32(1), np.int32(2)]
    slice_sizes = [2, 2, 2]
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

    # Input 5: 4D array, boolean
    operand = np.random.choice([True, False], size=(2, 3, 4, 5))
    start_indices = [np.int32(0), np.int32(0), np.int32(1), np.int32(1)]
    slice_sizes = [1, 2, 2, 3]
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

    # Input 6: 1D array, slice_sizes equal to full dimension
    operand = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    start_indices = [np.int32(0)]
    slice_sizes = [4]
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

    # Input 7: 2D array, slice_sizes of size 0
    operand = np.ones((4, 4), dtype=np.int32)
    start_indices = [np.int32(1), np.int32(1)]
    slice_sizes = [0, 2]
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

    # Input 8: Complex64 array
    operand = (
        np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    ).astype(np.complex64)
    start_indices = [np.int32(1), np.int32(0)]
    slice_sizes = [2, 1]
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

    # Input 9: 3D array, out of bounds start indices clamping test
    operand = np.zeros((3, 4, 5), dtype=np.float32)
    start_indices = [np.int32(10), np.int32(10), np.int32(10)]
    slice_sizes = [1, 2, 3]
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

    # Input 10: 2D array, int8, negative indices
    operand = np.arange(12, dtype=np.int8).reshape(3, 4)
    start_indices = [np.int32(-1), np.int32(-2)]
    slice_sizes = [1, 2]
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


generated_inputs["jax.lax.dynamic_slice_2"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_2'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_2'], lib="jax", suffix=2)
