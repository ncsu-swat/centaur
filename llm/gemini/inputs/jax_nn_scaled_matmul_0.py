
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.nn
import jax.numpy as jnp
from jax.interpreters import mlir
import numpy as np

try:
    _cudnn_func = jax.nn.scaled_matmul.__globals__["cudnn_scaled_matmul"]
    _p = _cudnn_func.__globals__["_scaled_matmul_p_wrapper"]

    def _scaled_matmul_cpu_rule(
        lhs, rhs, lhs_scales, rhs_scales, preferred_element_type=np.float32
    ):
        a_block_size = lhs.shape[-1] // lhs_scales.shape[-1]
        b_block_size = rhs.shape[-1] // rhs_scales.shape[-1]
        a_scaled = lhs * jnp.repeat(lhs_scales, a_block_size, axis=-1)
        b_scaled = rhs * jnp.repeat(rhs_scales, b_block_size, axis=-1)
        return jnp.einsum("BMK,BNK->BMN", a_scaled, b_scaled).astype(
            preferred_element_type
        )

    mlir.register_lowering(
        _p, mlir.lower_fun(_scaled_matmul_cpu_rule), platform="cpu"
    )
except Exception:
    pass


def scaled_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic minimal 3D case from documentation
    lhs = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    rhs = np.array([[[4.0, 5.0, 6.0]]], dtype=np.float32)
    lhs_scales = np.array([[[0.5]]], dtype=np.float32)
    rhs_scales = np.array([[[0.5]]], dtype=np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch size 2 with block-scale partition (K=16, K_scale=4)
    lhs = np.random.randn(2, 4, 16).astype(np.float32)
    rhs = np.random.randn(2, 8, 16).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, size=(2, 4, 4)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, size=(2, 8, 4)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative and positive values with single scale per row (K=32, K_scale=1)
    lhs = np.random.uniform(-5.0, 5.0, size=(4, 16, 32)).astype(np.float32)
    rhs = np.random.uniform(-5.0, 5.0, size=(4, 16, 32)).astype(np.float32)
    lhs_scales = np.random.uniform(0.5, 2.0, size=(4, 16, 1)).astype(np.float32)
    rhs_scales = np.random.uniform(0.5, 2.0, size=(4, 16, 1)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: preferred_element_type as float16 (K=64, K_scale=2)
    lhs = np.random.randn(1, 8, 64).astype(np.float32)
    rhs = np.random.randn(1, 4, 64).astype(np.float32)
    lhs_scales = np.random.uniform(0.5, 1.5, size=(1, 8, 2)).astype(np.float32)
    rhs_scales = np.random.uniform(0.5, 1.5, size=(1, 4, 2)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrices with finer scale granularity (K=128, K_scale=16)
    lhs = np.random.randn(3, 32, 128).astype(np.float32)
    rhs = np.random.randn(3, 64, 128).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, size=(3, 32, 16)).astype(
        np.float32
    )
    rhs_scales = np.random.uniform(0.1, 1.0, size=(3, 64, 16)).astype(
        np.float32
    )
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Full scale granularity (block size 1, K=4, K_scale=4)
    lhs = np.random.randn(2, 2, 4).astype(np.float32)
    rhs = np.random.randn(2, 2, 4).astype(np.float32)
    lhs_scales = np.ones((2, 2, 4), dtype=np.float32)
    rhs_scales = np.ones((2, 2, 4), dtype=np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Rectangular matrix dimensions (K=6, K_scale=3)
    lhs = np.random.randn(1, 3, 6).astype(np.float32)
    rhs = np.random.randn(1, 5, 6).astype(np.float32)
    lhs_scales = np.ones((1, 3, 3), dtype=np.float32)
    rhs_scales = np.ones((1, 5, 3), dtype=np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Batch size 2 with multiple rows (K=12, K_scale=6)
    lhs = np.random.uniform(-2.0, 2.0, size=(2, 6, 12)).astype(np.float32)
    rhs = np.random.uniform(-2.0, 2.0, size=(2, 3, 12)).astype(np.float32)
    lhs_scales = np.random.uniform(0.5, 1.0, size=(2, 6, 6)).astype(np.float32)
    rhs_scales = np.random.uniform(0.5, 1.0, size=(2, 3, 6)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Square M and N dimensions (K=20, K_scale=5)
    lhs = np.random.randn(1, 10, 20).astype(np.float32)
    rhs = np.random.randn(1, 10, 20).astype(np.float32)
    lhs_scales = np.random.uniform(0.2, 0.8, size=(1, 10, 5)).astype(np.float32)
    rhs_scales = np.random.uniform(0.2, 0.8, size=(1, 10, 5)).astype(
        np.float32
    )
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Vector-like batches (M=1, N=1, K=8, K_scale=8)
    lhs = np.random.randn(4, 1, 8).astype(np.float32)
    rhs = np.random.randn(4, 1, 8).astype(np.float32)
    lhs_scales = np.random.uniform(0.5, 2.0, size=(4, 1, 8)).astype(np.float32)
    rhs_scales = np.random.uniform(0.5, 2.0, size=(4, 1, 8)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Odd dimension lengths along K (K=15, K_scale=5)
    lhs = np.random.randn(2, 5, 15).astype(np.float32)
    rhs = np.random.randn(2, 7, 15).astype(np.float32)
    lhs_scales = np.random.uniform(0.1, 1.0, size=(2, 5, 5)).astype(np.float32)
    rhs_scales = np.random.uniform(0.1, 1.0, size=(2, 7, 5)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "lhs_scales": lhs_scales,
        "rhs_scales": rhs_scales,
        "preferred_element_type": np.dtype(np.float32),
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
