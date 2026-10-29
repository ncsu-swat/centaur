
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class DimensionNumbers(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 1, 2], dtype=dtype or np.int32)


def conv_transpose_inputs():
    list_of_inputs = []

    # 1. 2D conv transpose, NHWC format, VALID padding, unit strides
    lhs = np.random.randn(2, 8, 8, 3).astype(np.float32)
    rhs = np.random.randn(3, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1),
        "padding": "VALID",
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimensionNumbers(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": False,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. 2D conv transpose, NHWC format, SAME padding, stride 2, transposed kernel
    lhs = np.random.randn(1, 7, 7, 4).astype(np.float32)
    rhs = np.random.randn(3, 3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2),
        "padding": "SAME",
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimensionNumbers(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": True,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. 2D conv transpose, NCHW format, float64, dilated rhs
    lhs = np.random.randn(3, 2, 10, 10).astype(np.float64)
    rhs = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1),
        "padding": "VALID",
        "rhs_dilation": (2, 2),
        "dimension_numbers": DimensionNumbers(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": False,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. 2D conv transpose, NCHW format, asymmetric strides
    lhs = np.random.randn(2, 4, 6, 8).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 1),
        "padding": "SAME",
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimensionNumbers(("NCHW", "OIHW", "NCHW")),
        "transpose_kernel": True,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. 1D conv transpose, NHC format, VALID padding
    lhs = np.random.randn(2, 16, 4).astype(np.float32)
    rhs = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1,),
        "padding": "VALID",
        "rhs_dilation": (1,),
        "dimension_numbers": DimensionNumbers(("NHC", "HIO", "NHC")),
        "transpose_kernel": False,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. 1D conv transpose, NHC format, strided and dilated
    lhs = np.random.randn(1, 12, 2).astype(np.float32)
    rhs = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2,),
        "padding": "SAME",
        "rhs_dilation": (2,),
        "dimension_numbers": DimensionNumbers(("NHC", "HIO", "NHC")),
        "transpose_kernel": True,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. 3D conv transpose, NDHWC format, VALID padding
    lhs = np.random.randn(1, 4, 4, 4, 2).astype(np.float32)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 1, 1),
        "padding": "VALID",
        "rhs_dilation": (1, 1, 1),
        "dimension_numbers": DimensionNumbers(("NDHWC", "DHWIO", "NDHWC")),
        "transpose_kernel": False,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. 3D conv transpose, NDHWC format, SAME padding, strided, float64
    lhs = np.random.randn(2, 3, 3, 3, 2).astype(np.float64)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2, 2),
        "padding": "SAME",
        "rhs_dilation": (1, 1, 1),
        "dimension_numbers": DimensionNumbers(("NDHWC", "DHWIO", "NDHWC")),
        "transpose_kernel": True,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. 2D conv transpose, asymmetric kernel, NHWC format
    lhs = np.random.randn(2, 5, 9, 3).astype(np.float32)
    rhs = np.random.randn(3, 5, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (1, 2),
        "padding": "SAME",
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimensionNumbers(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": False,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
        "use_consistent_padding": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. 2D conv transpose, float64 with negative values, stride 2, VALID padding
    lhs = np.random.randn(1, 6, 6, 2).astype(np.float64)
    rhs = np.random.randn(2, 2, 2, 2).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "strides": (2, 2),
        "padding": "VALID",
        "rhs_dilation": (1, 1),
        "dimension_numbers": DimensionNumbers(("NHWC", "HWIO", "NHWC")),
        "transpose_kernel": True,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
        "use_consistent_padding": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.conv_transpose_1"] = conv_transpose_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_transpose_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_transpose_1'.")


check_valid('jax.lax.conv_transpose', generated_inputs['jax.lax.conv_transpose_1'], lib="jax", suffix=1)
