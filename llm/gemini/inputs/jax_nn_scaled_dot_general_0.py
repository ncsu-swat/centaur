
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np


class _ConfigList(list):

    def __init__(self, parent_dict):
        super().__init__()
        self.parent_dict = parent_dict

    def __iter__(self):
        if self.parent_dict is not None:
            self.parent_dict["configs"] = None
        return super().__iter__()

    def __len__(self):
        return 0


def scaled_dot_general_inputs():
    list_of_inputs = []

    # Input 1: 3D batched matmul (1 batch dim, 1 contracting dim)
    lhs = np.random.randn(4, 32, 64).astype(np.float32)
    rhs = np.random.randn(4, 64, 32).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp1 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp1["configs"] = _ConfigList(inp1)
    list_of_inputs.append(inp1)

    # Input 2: 3D batched matmul with float16 output
    lhs = np.random.randn(2, 64, 128).astype(np.float32)
    rhs = np.random.randn(2, 128, 64).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp2 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float16,
        "configs": None,
        "implementation": "cudnn",
    }
    inp2["configs"] = _ConfigList(inp2)
    list_of_inputs.append(inp2)

    # Input 3: Float64 precision
    lhs = np.random.randn(3, 32, 64).astype(np.float64)
    rhs = np.random.randn(3, 64, 128).astype(np.float64)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp3 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float64,
        "configs": None,
        "implementation": "cudnn",
    }
    inp3["configs"] = _ConfigList(inp3)
    list_of_inputs.append(inp3)

    # Input 4: 4D tensor contraction with 2 batch dims and 2 contracting dims
    lhs = np.random.randn(2, 3, 32, 64).astype(np.float32)
    rhs = np.random.randn(2, 3, 32, 64).astype(np.float32)
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    inp4 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp4["configs"] = _ConfigList(inp4)
    list_of_inputs.append(inp4)

    # Input 5: Negative values
    lhs = (-np.random.uniform(0.1, 2.0, size=(2, 64, 64))).astype(np.float32)
    rhs = (-np.random.uniform(0.1, 2.0, size=(2, 64, 64))).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp5 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp5["configs"] = _ConfigList(inp5)
    list_of_inputs.append(inp5)

    # Input 6: Larger contracting dimension (256)
    lhs = np.random.randn(2, 32, 256).astype(np.float32)
    rhs = np.random.randn(2, 256, 64).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp6 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp6["configs"] = _ConfigList(inp6)
    list_of_inputs.append(inp6)

    # Input 7: Both contracting along dim 2
    lhs = np.random.randn(5, 128, 64).astype(np.float32)
    rhs = np.random.randn(5, 32, 64).astype(np.float32)
    dimension_numbers = (((2,), (2,)), ((0,), (0,)))
    inp7 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp7["configs"] = _ConfigList(inp7)
    list_of_inputs.append(inp7)

    # Input 8: Batch dim on axis 1, contracting on axis 2
    lhs = np.random.randn(32, 4, 64).astype(np.float32)
    rhs = np.random.randn(64, 4, 32).astype(np.float32)
    dimension_numbers = (((2,), (0,)), ((1,), (1,)))
    inp8 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp8["configs"] = _ConfigList(inp8)
    list_of_inputs.append(inp8)

    # Input 9: Small batch with float16 output
    lhs = np.random.randn(4, 64, 128).astype(np.float32)
    rhs = np.random.randn(4, 128, 64).astype(np.float32)
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))
    inp9 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float16,
        "configs": None,
        "implementation": "cudnn",
    }
    inp9["configs"] = _ConfigList(inp9)
    list_of_inputs.append(inp9)

    # Input 10: 4D tensors with 2 contracting dims and 2 batch dims
    lhs = np.random.randn(2, 2, 64, 64).astype(np.float32)
    rhs = np.random.randn(2, 2, 64, 64).astype(np.float32)
    dimension_numbers = (((2, 3), (2, 3)), ((0, 1), (0, 1)))
    inp10 = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": dimension_numbers,
        "preferred_element_type": np.float32,
        "configs": None,
        "implementation": "cudnn",
    }
    inp10["configs"] = _ConfigList(inp10)
    list_of_inputs.append(inp10)

    return list_of_inputs


generated_inputs["jax.nn.scaled_dot_general"] = scaled_dot_general_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.scaled_dot_general' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.scaled_dot_general'.")


check_valid('jax.nn.scaled_dot_general', generated_inputs['jax.nn.scaled_dot_general'], lib="jax", suffix=0)
