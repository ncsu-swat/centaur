
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.mesh_utils import create_device_mesh

def dot_inputs():
    # Set up global mesh context to allow PartitionSpec() usage
    devices = jax.devices()
    mesh = Mesh(create_device_mesh((len(devices),), devices), ('x',))
    jax.set_mesh(mesh).__enter__()

    list_of_inputs = []

    # All inputs use homogeneous dimension_numbers to avoid numpy inhomogeneous shape errors.
    # We use 3D arrays of shape (B, M, K) and (B, K, N) with:
    # lhs_contracting = [2], rhs_contracting = [1]
    # lhs_batch = [0], rhs_batch = [0]
    # dimension_numbers = (([2], [1]), ([0], [0])) (homogeneous shape (2, 2, 1))
    # out_sharding is set to PartitionSpec() which is a tuple subclass of length 0.

    # Input 1: Standard float32, batch=1
    lhs = np.random.randn(1, 3, 4).astype(np.float32)
    rhs = np.random.randn(1, 4, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "default",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, batch=2, different sizes
    lhs = np.random.randn(2, 5, 3).astype(np.float32)
    rhs = np.random.randn(2, 3, 6).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "high",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, batch=1
    lhs = np.random.randn(1, 4, 4).astype(np.float64)
    rhs = np.random.randn(1, 4, 4).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "highest",
        "preferred_element_type": np.float64,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: int32, batch=1
    lhs = np.random.randint(-10, 10, size=(1, 3, 3)).astype(np.int32)
    rhs = np.random.randint(-10, 10, size=(1, 3, 3)).astype(np.int32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "default",
        "preferred_element_type": np.int32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64, batch=1
    lhs = (np.random.randn(1, 4, 2) + 1j * np.random.randn(1, 4, 2)).astype(np.complex64)
    rhs = (np.random.randn(1, 2, 4) + 1j * np.random.randn(1, 2, 4)).astype(np.complex64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "default",
        "preferred_element_type": np.complex64,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values, float32, batch=2
    lhs = np.random.uniform(-5.0, -1.0, size=(2, 3, 5)).astype(np.float32)
    rhs = np.random.uniform(-5.0, -1.0, size=(2, 5, 3)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "default",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large dimensions, float32, batch=1
    lhs = np.random.randn(1, 16, 32).astype(np.float32)
    rhs = np.random.randn(1, 32, 16).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "high",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small dimensions, float32, batch=4
    lhs = np.random.randn(4, 2, 2).astype(np.float32)
    rhs = np.random.randn(4, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "default",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High precision, float32, batch=1
    lhs = np.random.randn(1, 4, 3).astype(np.float32)
    rhs = np.random.randn(1, 3, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "high",
        "preferred_element_type": np.float32,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Highest precision, float64, batch=2
    lhs = np.random.randn(2, 3, 3).astype(np.float64)
    rhs = np.random.randn(2, 3, 3).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": "highest",
        "preferred_element_type": np.float64,
        "out_sharding": PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
