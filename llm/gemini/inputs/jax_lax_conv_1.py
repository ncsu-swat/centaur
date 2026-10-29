
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def conv_inputs():
    list_of_inputs = []

    # Input 1: 1D convolution, VALID padding, default precision, float32
    lhs = np.random.randn(1, 1, 8).astype(np.float32)
    rhs = np.random.randn(1, 1, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1],
        "padding": "VALID",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D convolution, SAME padding, stride 2, high precision
    lhs = np.random.randn(2, 3, 16).astype(np.float32)
    rhs = np.random.randn(4, 3, 4).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [2],
        "padding": "SAME",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution, standard 3x3 kernel, VALID padding, highest precision
    lhs = np.random.randn(2, 3, 10, 10).astype(np.float32)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1, 1],
        "padding": "VALID",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution, SAME padding, stride (2, 2)
    lhs = np.random.randn(4, 1, 14, 14).astype(np.float32)
    rhs = np.random.randn(2, 1, 5, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [2, 2],
        "padding": "SAME",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D convolution, asymmetric strides [1, 2], fastest precision
    lhs = np.random.randn(1, 4, 8, 16).astype(np.float32)
    rhs = np.random.randn(4, 4, 3, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1, 2],
        "padding": "VALID",
        "precision": "fastest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D convolution, VALID padding, stride [1, 1, 1]
    lhs = np.random.randn(1, 2, 5, 5, 5).astype(np.float32)
    rhs = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1, 1, 1],
        "padding": "VALID",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution, SAME padding, stride [2, 2, 2]
    lhs = np.random.randn(2, 1, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(3, 1, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [2, 2, 2],
        "padding": "SAME",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1x1 2D convolution (pointwise convolution)
    lhs = np.random.randn(4, 8, 12, 12).astype(np.float32)
    rhs = np.random.randn(16, 8, 1, 1).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1, 1],
        "padding": "VALID",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D convolution with negative values
    lhs = (np.random.randn(2, 2, 7, 9) * 5.0 - 2.5).astype(np.float32)
    rhs = (np.random.randn(3, 2, 3, 3) * 2.0 - 1.0).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [1, 1],
        "padding": "SAME",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D convolution with larger filter, VALID padding, stride 3
    lhs = np.random.randn(3, 2, 25).astype(np.float32)
    rhs = np.random.randn(5, 2, 7).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": [3],
        "padding": "VALID",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.conv_1"] = conv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.conv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.conv_1'.")


check_valid('jax.lax.conv', generated_inputs['jax.lax.conv_1'], lib="jax", suffix=1)
