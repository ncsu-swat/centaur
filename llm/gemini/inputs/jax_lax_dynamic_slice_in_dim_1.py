
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dynamic_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: 1D array, slicing from middle
    input_dict = {
        "operand": np.arange(10, dtype=np.int32),
        "start_index": 2,
        "slice_size": 4,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, slicing along axis 0
    input_dict = {
        "operand": np.random.randn(6, 8).astype(np.float32),
        "start_index": 1,
        "slice_size": 3,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, slicing along axis 1
    input_dict = {
        "operand": np.random.randn(5, 12).astype(np.float64),
        "start_index": 4,
        "slice_size": 5,
        "axis": 1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative start index with allow_negative_indices=True
    input_dict = {
        "operand": np.arange(20, dtype=np.int64),
        "start_index": -5,
        "slice_size": 3,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative axis (-1) on a 3D tensor
    input_dict = {
        "operand": np.random.uniform(size=(4, 5, 6)).astype(np.float32),
        "start_index": 2,
        "slice_size": 2,
        "axis": -1,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array with allow_negative_indices=False
    input_dict = {
        "operand": np.zeros((8, 6, 4), dtype=np.bool_),
        "start_index": 0,
        "slice_size": 4,
        "axis": 1,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: slice_size of 0
    input_dict = {
        "operand": np.arange(15, dtype=np.uint8),
        "start_index": 3,
        "slice_size": 0,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex tensor, 4D
    input_dict = {
        "operand": (
            np.random.randn(2, 3, 4, 5) + 1j * np.random.randn(2, 3, 4, 5)
        ).astype(np.complex64),
        "start_index": 1,
        "slice_size": 2,
        "axis": 2,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Full slice along axis (slice_size == dim length)
    input_dict = {
        "operand": np.random.randn(7, 3).astype(np.float32),
        "start_index": 0,
        "slice_size": 7,
        "axis": 0,
        "allow_negative_indices": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D tensor, slicing inner dimension
    input_dict = {
        "operand": np.ones((2, 2, 3, 4, 5), dtype=np.int16),
        "start_index": 2,
        "slice_size": 2,
        "axis": 4,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Out of bounds start index clipped behavior
    input_dict = {
        "operand": np.arange(10, dtype=np.float32),
        "start_index": 8,
        "slice_size": 4,
        "axis": 0,
        "allow_negative_indices": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.dynamic_slice_in_dim_1"] = (
    dynamic_slice_in_dim_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_slice_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_slice_in_dim_1'.")


check_valid('jax.lax.dynamic_slice_in_dim', generated_inputs['jax.lax.dynamic_slice_in_dim_1'], lib="jax", suffix=1)
