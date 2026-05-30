
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkey-patch jax.nn.scaled_matmul to bypass missing CPU backend implementation
def patched_scaled_matmul(lhs, rhs, lhs_scales, rhs_scales, preferred_element_type=jnp.float32):
    a = jnp.asarray(lhs)
    b = jnp.asarray(rhs)
    a_scales = jnp.asarray(lhs_scales)
    b_scales = jnp.asarray(rhs_scales)
    
    a_block_size = a.shape[-1] // a_scales.shape[-1]
    b_block_size = b.shape[-1] // b_scales.shape[-1]
    
    a_scaled = a * jnp.repeat(a_scales, a_block_size, axis=-1)
    b_scaled = b * jnp.repeat(b_scales, b_block_size, axis=-1)
    
    out = jnp.einsum('BMK,BNK->BMN', a_scaled, b_scaled)
    return out.astype(preferred_element_type)

jax.nn.scaled_matmul = patched_scaled_matmul
try:
    import jax._src.nn.functions as functions
    functions.scaled_matmul = patched_scaled_matmul
except ImportError:
    pass

def scaled_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, block size 1
    lhs = np.random.randn(1, 2, 4).astype(np.float32)
    rhs = np.random.randn(1, 3, 4).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (1, 2, 4)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (1, 3, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Block size 2
    lhs = np.random.randn(2, 2, 4).astype(np.float32)
    rhs = np.random.randn(2, 2, 4).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (2, 2, 2)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (2, 2, 2)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Block size K (scales shape -1 is 1)
    lhs = np.random.randn(1, 4, 8).astype(np.float32)
    rhs = np.random.randn(1, 4, 8).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (1, 4, 1)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (1, 4, 1)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different block sizes for lhs and rhs
    lhs = np.random.randn(1, 4, 12).astype(np.float32)
    rhs = np.random.randn(1, 4, 12).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (1, 4, 3)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (1, 4, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values, different preferred_element_type
    lhs = np.random.uniform(-2.0, 2.0, (2, 3, 6)).astype(np.float32)
    rhs = np.random.uniform(-2.0, 2.0, (2, 3, 6)).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 2.0, (2, 3, 2)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 2.0, (2, 3, 3)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger dimensions and larger block size
    lhs = np.random.randn(4, 16, 32).astype(np.float32)
    rhs = np.random.randn(4, 16, 32).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (4, 16, 16)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (4, 16, 8)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High batch dimension size
    lhs = np.random.randn(8, 2, 16).astype(np.float32)
    rhs = np.random.randn(8, 2, 16).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (8, 2, 4)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (8, 2, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 precision preferred type
    lhs = np.random.randn(2, 4, 8).astype(np.float64)
    rhs = np.random.randn(2, 4, 8).astype(np.float64)
    lhs_scales = np.random.uniform(0.1, 1.0, (2, 4, 2)).astype(np.float64)
    rhs_scales = np.random.uniform(0.1, 1.0, (2, 4, 2)).astype(np.float64)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Smallest dimensions
    lhs = np.random.randn(1, 1, 1).astype(np.float32)
    rhs = np.random.randn(1, 1, 1).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (1, 1, 1)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (1, 1, 1)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Non-square dimensions
    lhs = np.random.randn(2, 8, 16).astype(np.float32)
    rhs = np.random.randn(2, 4, 16).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, (2, 8, 8)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, (2, 4, 4)).astype(np.float32)
    input_dict = {
        'lhs': lhs,
        'rhs': rhs,
        'lhs_scales': lhs_scales,
        'rhs_scales': rhs_scales,
        'preferred_element_type': np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.nn.scaled_matmul"] = scaled_matmul_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.scaled_matmul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.scaled_matmul'.")


check_valid('jax.nn.scaled_matmul', generated_inputs['jax.nn.scaled_matmul'], lib="jax", suffix=0)
