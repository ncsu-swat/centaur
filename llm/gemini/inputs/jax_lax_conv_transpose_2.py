
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class DimensionTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array([0, 0], dtype=np.int32)

def conv_transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D standard convolution transpose (NCHW / OIHW)
    lhs = np.random.randn(2, 4, 8, 8).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [1, 1],
        "padding": [(0, 0), (0, 0)],
        "rhs_dilation": [1, 1],
        "dimension_numbers": DimensionTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D with stride 2 and use_consistent_padding=True
    lhs = np.random.randn(1, 3, 7, 7).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [2, 2],
        "padding": [(1, 1), (1, 1)],
        "rhs_dilation": [1, 1],
        "dimension_numbers": DimensionTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D NHWC format with dilation
    lhs = np.random.randn(2, 10, 10, 4).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [1, 1],
        "padding": [(0, 0), (0, 0)],
        "rhs_dilation": [2, 2],
        "dimension_numbers": DimensionTuple(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": False,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D convolution transpose (NCW / OIW)
    lhs = np.random.randn(4, 2, 16).astype(np.float32)
    rhs = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [1],
        "padding": [(0, 0)],
        "rhs_dilation": [1],
        "dimension_numbers": DimensionTuple(("NCW", "OIW", "NCW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D convolution transpose with stride 2 and float64
    lhs = np.random.randn(2, 4, 12).astype(np.float64)
    rhs = np.random.randn(4, 4, 4).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [2],
        "padding": [(1, 1)],
        "rhs_dilation": [1],
        "dimension_numbers": DimensionTuple(("NCW", "OIW", "NCW")),
        "transpose_kernel": True,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D convolution transpose (NCDHW / OIDHW)
    lhs = np.random.randn(1, 2, 4, 4, 4).astype(np.float32)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [1, 1, 1],
        "padding": [(0, 0), (0, 0), (0, 0)],
        "rhs_dilation": [1, 1, 1],
        "dimension_numbers": DimensionTuple(("NCDHW", "OIDHW", "NCDHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution transpose with strided dims and transpose_kernel=True
    lhs = np.random.randn(2, 2, 6, 6, 6).astype(np.float64)
    rhs = np.random.randn(2, 2, 3, 3, 3).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [2, 1, 2],
        "padding": [(1, 1), (0, 0), (1, 1)],
        "rhs_dilation": [1, 1, 1],
        "dimension_numbers": DimensionTuple(("NCDHW", "OIDHW", "NCDHW")),
        "transpose_kernel": True,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D NCHW with non-uniform strides and float64
    lhs = np.random.randn(3, 4, 9, 7).astype(np.float64)
    rhs = np.random.randn(4, 4, 3, 5).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [1, 2],
        "padding": [(2, 2), (1, 1)],
        "rhs_dilation": [1, 1],
        "dimension_numbers": DimensionTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D NCHW with asymmetric padding and larger strides
    lhs = np.random.randn(2, 3, 8, 8).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [2, 3],
        "padding": [(1, 2), (2, 1)],
        "rhs_dilation": [1, 1],
        "dimension_numbers": DimensionTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D NHWC with transpose_kernel=True and non-trivial dilation
    lhs = np.random.randn(1, 5, 5, 2).astype(np.float32)
    rhs = np.random.randn(3, 3, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": [2, 2],
        "padding": [(1, 1), (1, 1)],
        "rhs_dilation": [2, 1],
        "dimension_numbers": DimensionTuple(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": True,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_transpose_2"] = conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_transpose_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_transpose_2'.")


check_valid('jax.lax.conv_transpose', generated_inputs['jax.lax.conv_transpose_2'], lib="jax", suffix=2)
