
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D slicing
    input_dict = {
        'operand': np.arange(10, dtype=np.float32),
        'start_index': 2,
        'slice_size': 4,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Out of bound start index (clipped)
    input_dict = {
        'operand': np.arange(5, dtype=np.int32),
        'start_index': 4,
        'slice_size': 3,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, slicing along axis 1
    input_dict = {
        'operand': np.random.randn(3, 4).astype(np.float32),
        'start_index': 1,
        'slice_size': 2,
        'axis': 1,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, slicing along axis 2
    input_dict = {
        'operand': np.random.randn(2, 3, 5).astype(np.float32),
        'start_index': 2,
        'slice_size': 3,
        'axis': 2,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative start index
    input_dict = {
        'operand': np.arange(10, dtype=np.float32),
        'start_index': -3,
        'slice_size': 2,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Slicing with allow_negative_indices=False
    input_dict = {
        'operand': np.arange(10, dtype=np.float32),
        'start_index': 3,
        'slice_size': 4,
        'axis': 0,
        'allow_negative_indices': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Slicing 4D array along axis 0
    input_dict = {
        'operand': np.random.randn(5, 2, 2, 2).astype(np.float32),
        'start_index': 1,
        'slice_size': 3,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Slice size same as axis size
    input_dict = {
        'operand': np.random.randn(3, 3).astype(np.float32),
        'start_index': 0,
        'slice_size': 3,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Slice size 1
    input_dict = {
        'operand': np.arange(8, dtype=np.int32),
        'start_index': 5,
        'slice_size': 1,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean operand with 2D slicing
    input_dict = {
        'operand': np.ones((4, 4), dtype=bool),
        'start_index': 1,
        'slice_size': 2,
        'axis': 0,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array of uint8, axis 1
    input_dict = {
        'operand': np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        'start_index': 2,
        'slice_size': 2,
        'axis': 1,
        'allow_negative_indices': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.dynamic_slice_in_dim_1"] = dynamic_slice_in_dim_inputs()

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
