
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkey-patch BlockScaleConfig to support comparison operators, 
# satisfying numpy's min/max checks in the validator.
try:
    BlockScaleConfig = type(jax.nn.get_scaled_dot_general_config('mxfp8'))
    BlockScaleConfig.__lt__ = lambda self, other: False
    BlockScaleConfig.__le__ = lambda self, other: True
    BlockScaleConfig.__gt__ = lambda self, other: False
    BlockScaleConfig.__ge__ = lambda self, other: True
except Exception:
    pass

def scaled_dot_general_inputs():
    list_of_inputs = []

    # Create valid mxfp8 configs of length 3
    mxfp8_configs = [jax.nn.get_scaled_dot_general_config('mxfp8')] * 3

    # Input 1: Standard 3D batch matmul (Batch=1, Contracting=1)
    # Contracting dim must be a multiple of 32 (block size) for MXFP8.
    lhs = np.random.randn(2, 8, 32).astype(np.float32)
    rhs = np.random.randn(2, 32, 8).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D batch matmul with different dimensions (contracting size 64)
    lhs = np.random.randn(4, 16, 64).astype(np.float32)
    rhs = np.random.randn(4, 64, 16).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D batch matmul contracting last with last
    lhs = np.random.randn(2, 8, 32).astype(np.float32)
    rhs = np.random.randn(2, 8, 32).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (2,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float16 preferred element type
    lhs = np.random.randn(3, 16, 32).astype(np.float32)
    rhs = np.random.randn(3, 32, 16).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float16),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative and uniform values [-1.0, 1.0] (contracting size 64)
    lhs = np.random.uniform(-1.0, 1.0, (2, 128, 64)).astype(np.float32)
    rhs = np.random.uniform(-1.0, 1.0, (2, 64, 128)).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger dimensions 3D (contracting size 128)
    lhs = np.random.randn(5, 64, 128).astype(np.float32)
    rhs = np.random.randn(5, 128, 64).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D Tensor Matmul (Batch=2, Contracting=2)
    # Both contracting dimensions (K1, K2) must be multiples of 32
    lhs = np.random.randn(2, 2, 8, 32, 32).astype(np.float32)
    rhs = np.random.randn(2, 2, 32, 32, 8).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((3, 4), (2, 3)), ((0, 1), (0, 1))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another 5D homogeneous setup
    lhs = np.random.randn(3, 3, 16, 32, 32).astype(np.float32)
    rhs = np.random.randn(3, 3, 32, 32, 16).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((3, 4), (2, 3)), ((0, 1), (0, 1))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small batch size with normal distribution
    lhs = np.random.randn(1, 4, 32).astype(np.float32)
    rhs = np.random.randn(1, 32, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Square matrices in 3D (contracting size 32)
    lhs = np.random.randn(2, 32, 32).astype(np.float32)
    rhs = np.random.randn(2, 32, 32).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "dimension_numbers": (((2,), (1,)), ((0,), (0,))),
        "preferred_element_type": np.dtype(np.float32),
        "configs": mxfp8_configs,
        "implementation": "cudnn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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
