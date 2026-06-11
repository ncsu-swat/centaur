
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import jax
import copy

# Monkey-patch jax.lax.Precision to support comparison operators for NumPy min/max
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if hasattr(other, 'value') else NotImplemented

def jax_lax_dot_inputs():
    list_of_inputs = []

    # Input 1: Standard float32 batched dot
    lhs1 = np.random.randn(2, 3, 4).astype(np.float32)
    rhs1 = np.random.randn(2, 4, 5).astype(np.float32)
    input_dict1 = {
        "lhs": lhs1,
        "rhs": rhs1,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: High precision float32 batched dot
    lhs2 = np.random.randn(3, 5, 2).astype(np.float32)
    rhs2 = np.random.randn(3, 2, 6).astype(np.float32)
    input_dict2 = {
        "lhs": lhs2,
        "rhs": rhs2,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Highest precision float64 batched dot
    lhs3 = np.random.randn(4, 2, 3).astype(np.float64)
    rhs3 = np.random.randn(4, 3, 2).astype(np.float64)
    input_dict3 = {
        "lhs": lhs3,
        "rhs": rhs3,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float64,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Transposed batching coordinates
    lhs4 = np.random.randn(2, 4, 3).astype(np.float32)
    rhs4 = np.random.randn(3, 2, 5).astype(np.float32)
    input_dict4 = {
        "lhs": lhs4,
        "rhs": rhs4,
        "dimension_numbers": (([2], [0]), ([0], [1])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Alternative batch and contracting axes alignment
    lhs5 = np.random.randn(5, 2, 3).astype(np.float32)
    rhs5 = np.random.randn(2, 5, 4).astype(np.float32)
    input_dict5 = {
        "lhs": lhs5,
        "rhs": rhs5,
        "dimension_numbers": (([1], [0]), ([0], [1])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Integer batched multiplication (int32)
    lhs6 = np.random.randint(-10, 10, size=(2, 3, 2)).astype(np.int32)
    rhs6 = np.random.randint(-10, 10, size=(2, 2, 4)).astype(np.int32)
    input_dict6 = {
        "lhs": lhs6,
        "rhs": rhs6,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.int32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Complex64 batched multiplication
    lhs7 = (np.random.randn(2, 3, 3) + 1j * np.random.randn(2, 3, 3)).astype(np.complex64)
    rhs7 = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(np.complex64)
    input_dict7 = {
        "lhs": lhs7,
        "rhs": rhs7,
        "dimension_numbers": (([1], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.complex64,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Higher dimensions (4D) with 1 batch, 1 contract
    lhs8 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    rhs8 = np.random.randn(2, 3, 5, 6).astype(np.float32)
    input_dict8 = {
        "lhs": lhs8,
        "rhs": rhs8,
        "dimension_numbers": (([3], [2]), ([0], [0])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Negative float values
    lhs9 = np.random.uniform(-5.0, -1.0, size=(2, 2, 2)).astype(np.float32)
    rhs9 = np.random.uniform(-5.0, -1.0, size=(2, 2, 2)).astype(np.float32)
    input_dict9 = {
        "lhs": lhs9,
        "rhs": rhs9,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Mixed precision accumulation to float64
    lhs10 = np.random.randn(3, 2, 4).astype(np.float32)
    rhs10 = np.random.randn(3, 4, 2).astype(np.float32)
    input_dict10 = {
        "lhs": lhs10,
        "rhs": rhs10,
        "dimension_numbers": (([2], [1]), ([0], [0])),
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float64,
        "out_sharding": jax.sharding.PartitionSpec()
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["jax.lax.dot_2"] = jax_lax_dot_inputs()

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
