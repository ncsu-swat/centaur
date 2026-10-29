
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def slice_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, basic slicing
    operand = np.arange(10, dtype=np.int32)
    start_indices = [2]
    limit_indices = [7]
    strides = [1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 2: 2D float32 array with non-unit strides
    operand = np.random.randn(8, 6).astype(np.float32)
    start_indices = [1, 0]
    limit_indices = [7, 6]
    strides = [2, 3]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 3: 2D float64 array, full slice
    operand = np.random.randn(4, 4).astype(np.float64)
    start_indices = [0, 0]
    limit_indices = [4, 4]
    strides = [1, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 4: 3D int64 array, interior slice
    operand = np.arange(60, dtype=np.int64).reshape((3, 4, 5))
    start_indices = [1, 1, 2]
    limit_indices = [3, 3, 5]
    strides = [1, 1, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 5: 3D boolean array with stride > 1
    operand = np.random.choice([True, False], size=(6, 8, 4))
    start_indices = [0, 2, 1]
    limit_indices = [6, 8, 4]
    strides = [2, 2, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 6: 4D float32 array
    operand = np.random.randn(2, 5, 6, 3).astype(np.float32)
    start_indices = [0, 1, 0, 1]
    limit_indices = [2, 4, 5, 3]
    strides = [1, 1, 2, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 7: 1D array with zero-length slice (start == limit)
    operand = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    start_indices = [5]
    limit_indices = [5]
    strides = [1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 8: 2D complex64 array
    operand = (
        np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
    ).astype(np.complex64)
    start_indices = [1, 2]
    limit_indices = [4, 5]
    strides = [1, 2]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 9: 5D uint8 array
    operand = np.random.randint(0, 256, size=(2, 3, 4, 3, 2), dtype=np.uint8)
    start_indices = [0, 0, 1, 0, 0]
    limit_indices = [2, 2, 3, 3, 2]
    strides = [1, 1, 1, 1, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 10: 1D int16 array with large stride
    operand = np.arange(100, dtype=np.int16)
    start_indices = [10]
    limit_indices = [90]
    strides = [15]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    # Input 11: 3D float16 array with single element slice along one axis
    operand = np.random.randn(4, 4, 4).astype(np.float16)
    start_indices = [2, 0, 1]
    limit_indices = [3, 4, 3]
    strides = [1, 2, 1]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.slice_1"] = slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.slice_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.slice_1'.")


check_valid('jax.lax.slice', generated_inputs['jax.lax.slice_1'], lib="jax", suffix=1)
