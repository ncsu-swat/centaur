
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np

_mesh = Mesh(np.array([jax.devices("cpu")[0]]), ("x",))
jax.set_mesh(_mesh)


def dot_inputs():
    list_of_inputs = []

    # Input 1: Batched matrix multiplication (1 batch dim, 1 contracting dim), float32
    lhs = np.random.randn(2, 3, 4).astype(np.float32)
    rhs = np.random.randn(2, 4, 5).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 2: Batched matrix multiplication with float32, 'high' precision
    lhs = np.random.randn(4, 2, 3).astype(np.float32)
    rhs = np.random.randn(4, 3, 2).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "high"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 3: Negative float32 values, 'highest' precision
    lhs = -np.random.uniform(1.0, 5.0, size=(3, 5, 2)).astype(np.float32)
    rhs = np.random.uniform(-4.0, -1.0, size=(3, 2, 4)).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "highest"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 4: Different dimension ordering for contracting/batch dims, 'fastest' precision
    lhs = np.random.randn(3, 2, 4).astype(np.float32)
    rhs = np.random.randn(4, 2, 5).astype(np.float32)
    dimension_numbers = (((2,), (0,)), ((1,), (1,)))
    precision = "fastest"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 5: 2 batch dims, 2 contracting dims, 'high' precision
    lhs = np.random.randn(2, 3, 2, 4, 5).astype(np.float32)
    rhs = np.random.randn(2, 3, 4, 5, 3).astype(np.float32)
    dimension_numbers = (((3, 4), (2, 3)), ((0, 1), (0, 1)))
    precision = "high"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 6: Integer dot product (int32)
    lhs = np.random.randint(-10, 10, size=(2, 4, 3)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
    preferred_element_type = np.dtype(np.int32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 7: Complex numbers dot product (complex64)
    lhs = (
        np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)
    ).astype(np.complex64)
    rhs = (
        np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)
    ).astype(np.complex64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
    preferred_element_type = np.dtype(np.complex64)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 8: Batch size 1 with float32
    lhs = np.random.randn(1, 3, 3).astype(np.float32)
    rhs = np.random.randn(1, 3, 2).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "highest"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 9: Positive integers (int32)
    lhs = np.random.randint(1, 10, size=(2, 2, 2)).astype(np.int32)
    rhs = np.random.randint(1, 10, size=(2, 2, 2)).astype(np.int32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "default"
    preferred_element_type = np.dtype(np.int32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    # Input 10: Larger batched matrices with float32
    lhs = np.random.randn(8, 16, 8).astype(np.float32)
    rhs = np.random.randn(8, 8, 12).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = "high"
    preferred_element_type = np.dtype(np.float32)
    out_sharding = PartitionSpec()
    list_of_inputs.append(
        copy.deepcopy(
            {
                "lhs": lhs,
                "rhs": rhs,
                "dimension_numbers": dimension_numbers,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
                "out_sharding": out_sharding,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.dot_1"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_1'.")


check_valid('jax.lax.dot', generated_inputs['jax.lax.dot_1'], lib="jax", suffix=1)
