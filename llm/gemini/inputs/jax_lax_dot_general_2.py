
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
from jax.sharding import Mesh, PartitionSpec

# Create and enter a global mesh context so PartitionSpec can be canonicalized
devices = jax.devices()
mesh = Mesh(np.array(devices[:1]), ('x',))
mesh.__enter__()

def dot_general_inputs():
    list_of_inputs = []

    # Case 1: Batch matrix multiplication with float32 (Type A)
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (0, 0)
    preferred_element_type = np.dtype('float32')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 2: Batch matrix multiplication with float64 (Type A)
    lhs = np.random.randn(3, 2, 5).astype(np.float64)
    rhs = np.random.randn(3, 5, 2).astype(np.float64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (1, 1)
    preferred_element_type = np.dtype('float64')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 3: Batch matrix multiplication with int32 (Type A)
    lhs = np.random.randint(-10, 10, size=(2, 4, 3)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (0, 0)
    preferred_element_type = np.dtype('int32')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 4: 4D tensor contraction with 2 contracting and 2 batch dims (Type B)
    lhs = np.random.randn(2, 2, 3, 3).astype(np.float32)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float32)
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    precision = (2, 2)
    preferred_element_type = np.dtype('float32')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 5: Outer product with float32 (Type C)
    lhs = np.random.randn(3, 3).astype(np.float32)
    rhs = np.random.randn(2, 2).astype(np.float32)
    dimension_numbers = (((), ()), ((), ()))
    precision = (0, 0)
    preferred_element_type = np.dtype('float32')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 6: Batch matrix multiplication with float16 (Type A)
    lhs = np.random.randn(2, 5, 2).astype(np.float16)
    rhs = np.random.randn(2, 2, 5).astype(np.float16)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (0, 0)
    preferred_element_type = np.dtype('float16')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 7: 4D tensor contraction with float64 (Type B)
    lhs = np.random.randn(2, 3, 2, 2).astype(np.float64)
    rhs = np.random.randn(2, 3, 2, 2).astype(np.float64)
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    precision = (1, 1)
    preferred_element_type = np.dtype('float64')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 8: Batch matrix multiplication with complex64 (Type A)
    real_lhs = np.random.randn(2, 3, 3).astype(np.float32)
    imag_lhs = np.random.randn(2, 3, 3).astype(np.float32)
    real_rhs = np.random.randn(2, 3, 3).astype(np.float32)
    imag_rhs = np.random.randn(2, 3, 3).astype(np.float32)
    lhs = real_lhs + 1j * imag_lhs
    rhs = real_rhs + 1j * imag_rhs
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (0, 0)
    preferred_element_type = np.dtype('complex64')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 9: Outer product with float64 (Type C)
    lhs = np.random.randn(4, 2).astype(np.float64)
    rhs = np.random.randn(2, 4).astype(np.float64)
    dimension_numbers = (((), ()), ((), ()))
    precision = (2, 2)
    preferred_element_type = np.dtype('float64')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    # Case 10: Batch matrix multiplication with different dimensions (Type A)
    lhs = np.random.randn(4, 2, 6).astype(np.float32)
    rhs = np.random.randn(4, 6, 3).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = (0, 0)
    preferred_element_type = np.dtype('float32')
    out_sharding = PartitionSpec()
    list_of_inputs.append({
        'lhs': lhs,
        'rhs': rhs,
        'dimension_numbers': dimension_numbers,
        'precision': precision,
        'preferred_element_type': preferred_element_type,
        'out_sharding': out_sharding
    })

    return list_of_inputs

generated_inputs["jax.lax.dot_general_2"] = dot_general_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_general_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_general_2'.")


check_valid('jax.lax.dot_general', generated_inputs['jax.lax.dot_general_2'], lib="jax", suffix=2)
