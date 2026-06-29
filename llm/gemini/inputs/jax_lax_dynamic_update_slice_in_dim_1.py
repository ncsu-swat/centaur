
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dynamic_update_slice_in_dim_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array
    list_of_inputs.append({
        'operand': np.zeros(6, dtype=np.float32),
        'update': np.ones(3, dtype=np.float32),
        'start_index': 2,
        'axis': 0,
        'allow_negative_indices': True
    })

    # Input 2: 1D array with index that overflows (will be clamped/adjusted by JAX)
    list_of_inputs.append({
        'operand': np.zeros(6, dtype=np.float32),
        'update': np.ones(3, dtype=np.float32),
        'start_index': 5,
        'axis': 0,
        'allow_negative_indices': True
    })

    # Input 3: 2D array, update along axis 0
    list_of_inputs.append({
        'operand': np.zeros((4, 4), dtype=np.float32),
        'update': np.ones((2, 4), dtype=np.float32),
        'start_index': 1,
        'axis': 0,
        'allow_negative_indices': True
    })

    # Input 4: 2D array, update along axis 1, smaller update on non-axis dimension
    list_of_inputs.append({
        'operand': np.zeros((4, 4), dtype=np.float32),
        'update': np.ones((3, 2), dtype=np.float32),
        'start_index': 1,
        'axis': 1,
        'allow_negative_indices': False
    })

    # Input 5: Negative start index
    list_of_inputs.append({
        'operand': np.zeros(10, dtype=np.float32),
        'update': np.ones(4, dtype=np.float32),
        'start_index': -3,
        'axis': 0,
        'allow_negative_indices': True
    })

    # Input 6: 3D array, axis=2
    list_of_inputs.append({
        'operand': np.zeros((3, 4, 5), dtype=np.float32),
        'update': np.ones((2, 2, 2), dtype=np.float32),
        'start_index': 2,
        'axis': 2,
        'allow_negative_indices': True
    })

    # Input 7: Float64 types, larger dimension
    list_of_inputs.append({
        'operand': np.zeros((100, 100), dtype=np.float64),
        'update': np.ones((10, 10), dtype=np.float64),
        'start_index': 50,
        'axis': 0,
        'allow_negative_indices': False
    })

    # Input 8: Integer types (int32), axis=1
    list_of_inputs.append({
        'operand': np.zeros((5, 5), dtype=np.int32),
        'update': np.ones((2, 2), dtype=np.int32),
        'start_index': 2,
        'axis': 1,
        'allow_negative_indices': True
    })

    # Input 9: 4D arrays, complex types (complex64)
    list_of_inputs.append({
        'operand': np.zeros((2, 3, 4, 5), dtype=np.complex64),
        'update': np.ones((2, 2, 2, 2), dtype=np.complex64),
        'start_index': 1,
        'axis': 3,
        'allow_negative_indices': True
    })

    # Input 10: Boolean types
    list_of_inputs.append({
        'operand': np.zeros((4, 4), dtype=bool),
        'update': np.ones((2, 2), dtype=bool),
        'start_index': 0,
        'axis': 0,
        'allow_negative_indices': False
    })

    return list_of_inputs

generated_inputs["jax.lax.dynamic_update_slice_in_dim_1"] = dynamic_update_slice_in_dim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_in_dim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_in_dim_1'.")


check_valid('jax.lax.dynamic_update_slice_in_dim', generated_inputs['jax.lax.dynamic_update_slice_in_dim_1'], lib="jax", suffix=1)
