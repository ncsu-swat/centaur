
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
from jax.sharding import Mesh, PartitionSpec
import numpy as np


def dot_inputs():
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("x",))
    jax.set_mesh(mesh)

    list_of_inputs = []

    # Input 1: Batched matrix multiplication (float32)
    lhs = np.random.randn(2, 4, 3).astype(np.float32)
    rhs = np.random.randn(2, 3, 5).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("default", "default")
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

    # Input 2: Batched matrix multiplication (float64)
    lhs = np.random.randn(3, 5, 8).astype(np.float64)
    rhs = np.random.randn(3, 8, 4).astype(np.float64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("highest", "highest")
    preferred_element_type = np.dtype(np.float64)
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

    # Input 3: Batched contraction on matching last dimensions
    lhs = np.random.randn(4, 6, 2).astype(np.float32)
    rhs = np.random.randn(4, 7, 2).astype(np.float32)
    dimension_numbers = (((2,), (2,)), ((0,), (0,)))
    precision = ("high", "high")
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

    # Input 4: Batch dimension at index 1 with negative values
    lhs = np.random.uniform(-5.0, -1.0, size=(3, 2, 4)).astype(np.float32)
    rhs = np.random.uniform(-3.0, 3.0, size=(5, 2, 4)).astype(np.float32)
    dimension_numbers = (((2,), (2,)), ((1,), (1,)))
    precision = ("default", "default")
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

    # Input 5: 4D tensors with 2 batch dims and 2 contracting dims
    lhs = np.random.randn(2, 3, 4, 5).astype(np.float32)
    rhs = np.random.randn(2, 3, 4, 5).astype(np.float32)
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    precision = ("default", "high")
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

    # Input 6: Integer dot product with negative values
    lhs = np.random.randint(-10, 10, size=(3, 4, 4)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(3, 4, 4)).astype(np.int32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("default", "default")
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

    # Input 7: Complex-valued batched dot product
    lhs = (
        np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)
    ).astype(np.complex64)
    rhs = (
        np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)
    ).astype(np.complex64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("default", "default")
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

    # Input 8: Upcasting precision from float32 to float64
    lhs = np.random.randn(2, 5, 6).astype(np.float32)
    rhs = np.random.randn(2, 6, 4).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("high", "highest")
    preferred_element_type = np.dtype(np.float64)
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

    # Input 9: Contracting non-contiguous dimensions
    lhs = np.random.randn(5, 2, 6).astype(np.float32)
    rhs = np.random.randn(6, 2, 3).astype(np.float32)
    dimension_numbers = (((2,), (0,)), ((1,), (1,)))
    precision = ("default", "default")
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

    # Input 10: Larger batched contraction
    lhs = np.random.randn(8, 12, 16).astype(np.float32)
    rhs = np.random.randn(8, 16, 8).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    precision = ("default", "high")
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


generated_inputs["jax.lax.dot_2"] = dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dot_2'.")


check_valid('jax.lax.dot', generated_inputs['jax.lax.dot_2'], lib="jax", suffix=2)
