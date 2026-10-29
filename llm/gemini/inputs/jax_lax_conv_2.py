
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0], dtype=np.int32)


def conv_inputs():
    list_of_inputs = []

    # Input 1: 1D convolution, VALID padding, default precision, float32
    lhs = np.random.randn(2, 3, 10).astype(np.float32)
    rhs = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1,),
        "padding": "VALID",
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D convolution, SAME padding, stride 2, float64
    lhs = np.random.randn(1, 2, 16).astype(np.float64)
    rhs = np.random.randn(2, 2, 4).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2,),
        "padding": "SAME",
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution, VALID padding, stride (1, 1), float32
    lhs = np.random.randn(4, 1, 8, 8).astype(np.float32)
    rhs = np.random.randn(8, 1, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1),
        "padding": "VALID",
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution, SAME padding, stride (2, 2), float32
    lhs = np.random.randn(2, 4, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 4, 5, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2, 2),
        "padding": "SAME",
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D convolution, asymmetric strides, float64
    lhs = np.random.randn(3, 2, 12, 14).astype(np.float64)
    rhs = np.random.randn(4, 2, 3, 2).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 2),
        "padding": "VALID",
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D convolution, VALID padding, stride (1, 1, 1), float32
    lhs = np.random.randn(1, 2, 6, 6, 6).astype(np.float32)
    rhs = np.random.randn(3, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1, 1),
        "padding": "VALID",
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution, SAME padding, stride (2, 1, 2), float32
    lhs = np.random.randn(2, 1, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(2, 1, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2, 1, 2),
        "padding": "SAME",
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D convolution with negative values, SAME padding
    lhs = -np.ones((2, 3, 7), dtype=np.float32)
    rhs = 0.5 * np.ones((1, 3, 3), dtype=np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1,),
        "padding": "SAME",
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D convolution with 1x1 kernel (pointwise conv), float32
    lhs = np.random.randn(4, 16, 7, 7).astype(np.float32)
    rhs = np.random.randn(32, 16, 1, 1).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1),
        "padding": "VALID",
        "precision": PrecisionTuple(
            (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D convolution, larger kernel and batch size
    lhs = np.random.randn(8, 3, 20, 20).astype(np.float32)
    rhs = np.random.randn(16, 3, 7, 7).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (3, 3),
        "padding": "SAME",
        "precision": PrecisionTuple(
            (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)
        ),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.conv_2"] = conv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_2'.")


check_valid('jax.lax.conv', generated_inputs['jax.lax.conv_2'], lib="jax", suffix=2)
