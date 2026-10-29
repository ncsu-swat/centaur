
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

class DimTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array(list(self), dtype=object)

def conv_transpose_inputs():
    list_of_inputs = []

    # Input 1: 1D convolution transpose, NCW format
    lhs = np.random.randn(1, 2, 8).astype(np.float32)
    rhs = np.random.randn(2, 2, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1,),
        "padding": ((0, 0),),
        "rhs_dilation": (1,),
        "dimension_numbers": DimTuple(("NCW", "OIW", "NCW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D convolution transpose, NWC format, stride 2, transposed kernel
    lhs = np.random.randn(2, 16, 4).astype(np.float32)
    rhs = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2,),
        "padding": ((1, 1),),
        "rhs_dilation": (1,),
        "dimension_numbers": DimTuple(("NWC", "WIO", "NWC")),
        "transpose_kernel": True,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution transpose, NCHW, consistent padding
    lhs = np.random.randn(1, 3, 7, 7).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1),
        "padding": ((0, 0), (0, 0)),
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution transpose, NHWC, stride 2, transposed kernel
    lhs = np.random.randn(2, 8, 8, 4).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2),
        "padding": ((1, 1), (1, 1)),
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimTuple(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": True,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D convolution transpose with dilation on rhs
    lhs = np.random.randn(1, 2, 10, 10).astype(np.float32)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1),
        "padding": ((1, 1), (1, 1)),
        "rhs_dilation": (2, 2),
        "dimension_numbers": DimTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D convolution transpose with asymmetric strides and padding
    lhs = np.random.randn(4, 3, 12, 16).astype(np.float32)
    rhs = np.random.randn(3, 3, 2, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 2),
        "padding": ((0, 1), (2, 1)),
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution transpose, NCDHW format
    lhs = np.random.randn(1, 2, 4, 6, 6).astype(np.float32)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1, 1),
        "padding": ((0, 0), (0, 0), (0, 0)),
        "rhs_dilation": (1, 1, 1),
        "dimension_numbers": DimTuple(("NCDHW", "OIDHW", "NCDHW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D convolution transpose, NDHWC format, stride 2
    lhs = np.random.randn(2, 4, 4, 4, 2).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2, 2),
        "padding": ((1, 1), (1, 1), (1, 1)),
        "rhs_dilation": (1, 1, 1),
        "dimension_numbers": DimTuple(("NDHWC", "DHWIO", "NDHWC")),
        "transpose_kernel": True,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D convolution transpose with float64 inputs
    lhs = np.random.randn(2, 2, 5, 5).astype(np.float64)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 1),
        "padding": ((0, 0), (1, 1)),
        "rhs_dilation": (1, 2),
        "dimension_numbers": DimTuple(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D convolution transpose with use_consistent_padding=True
    lhs = np.random.randn(3, 1, 10).astype(np.float32)
    rhs = np.random.randn(1, 1, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1,),
        "padding": ((1, 1),),
        "rhs_dilation": (1,),
        "dimension_numbers": DimTuple(("NCW", "OIW", "NCW")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D convolution transpose with 1x1 kernel
    lhs = np.random.randn(1, 4, 4, 8).astype(np.float32)
    rhs = np.random.randn(1, 1, 8, 8).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2),
        "padding": ((0, 0), (0, 0)),
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimTuple(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_transpose_3"] = conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_transpose_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_transpose_3'.")


check_valid('jax.lax.conv_transpose', generated_inputs['jax.lax.conv_transpose_3'], lib="jax", suffix=3)
