
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
from jax.sharding import Mesh, PartitionSpec

# Set up a global mesh to allow PartitionSpec canonicalization
devices = jax.devices()
mesh = Mesh(np.array(devices), ('x',))
jax.set_mesh(mesh)

def dot_general_inputs():
    list_of_inputs = []

    # Input 1: Basic 3D with 1 batch and 1 contracting dim
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 2: High precision
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "high"
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

    # Input 3: Highest precision with float64
    lhs = np.random.randn(2, 3, 4).astype(np.float64)
    rhs = np.random.randn(2, 4, 5).astype(np.float64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "highest"
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

    # Input 4: Negative integer values
    lhs = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(2, 4, 5)).astype(np.int32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 5: Different dimensions
    lhs = np.random.randn(5, 2, 3).astype(np.float32)
    rhs = np.random.randn(5, 3, 4).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 6: Large square dimensions
    lhs = np.random.randn(1, 10, 10).astype(np.float32)
    rhs = np.random.randn(1, 10, 10).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 7: Float16 datatype
    lhs = np.random.randn(3, 8, 16).astype(np.float16)
    rhs = np.random.randn(3, 16, 8).astype(np.float16)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 8: Complex64 numbers
    lhs = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex64)
    rhs = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

    # Input 9: High precision with different dimensions
    lhs = np.random.randn(4, 5, 2).astype(np.float32)
    rhs = np.random.randn(4, 2, 6).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "high"
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

    # Input 10: Zero-sized dimensions
    lhs = np.random.randn(2, 3, 0).astype(np.float32)
    rhs = np.random.randn(2, 0, 4).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
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

generated_inputs["jax.lax.dot_general_1"] = dot_general_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_general_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_general_1'.")


check_valid('jax.lax.dot_general', generated_inputs['jax.lax.dot_general_1'], lib="jax", suffix=1)
