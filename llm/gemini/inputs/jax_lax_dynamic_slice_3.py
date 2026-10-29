
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax
import jax.lax


def dynamic_slice_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, standard slice
    operand = np.arange(10, dtype=np.float32)
    start_indices = np.array([2], dtype=np.int32)
    slice_sizes = (4,)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 2: 2D int32 array, 2D slice
    operand = np.arange(20, dtype=np.int32).reshape(4, 5)
    start_indices = np.array([1, 2], dtype=np.int32)
    slice_sizes = (2, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 3: 3D float64 array, allow_negative_indices False
    operand = np.random.randn(3, 4, 5).astype(np.float64)
    start_indices = np.array([0, 1, 2], dtype=np.int64)
    slice_sizes = (2, 2, 2)
    allow_negative_indices = False
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 4: 2D with negative start indices
    operand = np.arange(12, dtype=np.float32).reshape(3, 4)
    start_indices = np.array([-2, -3], dtype=np.int32)
    slice_sizes = (2, 2)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 5: 4D boolean array
    operand = np.random.choice([True, False], size=(2, 3, 4, 5))
    start_indices = np.array([1, 0, 2, 1], dtype=np.int32)
    slice_sizes = (1, 2, 2, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 6: Zero slice sizes along one dimension
    operand = np.ones((5, 5), dtype=np.float32)
    start_indices = np.array([1, 1], dtype=np.int32)
    slice_sizes = (0, 3)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 7: Full size slice
    operand = np.arange(16, dtype=np.int64).reshape(4, 4)
    start_indices = np.array([0, 0], dtype=np.int64)
    slice_sizes = (4, 4)
    allow_negative_indices = False
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 8: Complex64 array
    operand = (
        np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
    ).astype(np.complex64)
    start_indices = np.array([2, 3], dtype=np.int32)
    slice_sizes = (3, 2)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 9: 5D int16 array
    operand = np.ones((2, 2, 3, 3, 2), dtype=np.int16)
    start_indices = np.array([0, 1, 1, 0, 1], dtype=np.int32)
    slice_sizes = (1, 1, 2, 2, 1)
    allow_negative_indices = False
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    # Input 10: 1D array with slice exceeding bound (clamped by dynamic_slice)
    operand = np.arange(8, dtype=np.uint8)
    start_indices = np.array([6], dtype=np.int32)
    slice_sizes = (4,)
    allow_negative_indices = True
    list_of_inputs.append(
        {
            "operand": operand,
            "start_indices": start_indices,
            "slice_sizes": slice_sizes,
            "allow_negative_indices": allow_negative_indices,
        }
    )

    return list_of_inputs


generated_inputs["jax.lax.dynamic_slice_3"] = dynamic_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_3'.")


check_valid('jax.lax.dynamic_slice', generated_inputs['jax.lax.dynamic_slice_3'], lib="jax", suffix=3)
