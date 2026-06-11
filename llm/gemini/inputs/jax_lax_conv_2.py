
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.lax.Precision to support comparisons,
# preventing np.min/np.max from raising TypeErrors during signature validation.
jax.lax.Precision.__lt__ = lambda self, other: self.value < other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__le__ = lambda self, other: self.value <= other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__gt__ = lambda self, other: self.value > other.value if hasattr(other, 'value') else NotImplemented
jax.lax.Precision.__ge__ = lambda self, other: self.value >= other.value if hasattr(other, 'value') else NotImplemented

def conv_inputs():
    list_of_inputs = []

    # Input 1: 1D convolution, VALID padding, float32
    lhs = np.random.randn(2, 3, 10).astype(np.float32)
    rhs = np.random.randn(4, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1,),
        "padding": "VALID",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D convolution, SAME padding, stride 2, float32
    lhs = np.random.randn(1, 1, 16).astype(np.float32)
    rhs = np.random.randn(1, 1, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2,),
        "padding": "SAME",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D convolution, SAME padding, float32
    lhs = np.random.randn(2, 3, 16, 16).astype(np.float32)
    rhs = np.random.randn(8, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1),
        "padding": "SAME",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D convolution, VALID padding, stride 2, float64
    lhs = np.random.randn(1, 4, 28, 28).astype(np.float64)
    rhs = np.random.randn(16, 4, 5, 5).astype(np.float64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2, 2),
        "padding": "VALID",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.float64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D convolution, non-square stride (1, 2)
    lhs = np.random.randn(4, 2, 32, 64).astype(np.float32)
    rhs = np.random.randn(4, 2, 3, 5).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 2),
        "padding": "SAME",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D convolution, SAME padding
    lhs = np.random.randn(1, 2, 8, 8, 8).astype(np.float32)
    rhs = np.random.randn(4, 2, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1, 1),
        "padding": "SAME",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D convolution, VALID padding, stride 2
    lhs = np.random.randn(2, 1, 16, 16, 16).astype(np.float32)
    rhs = np.random.randn(2, 1, 3, 3, 3).astype(np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (2, 2, 2),
        "padding": "VALID",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D convolution, float16
    lhs = np.random.randn(1, 2, 32).astype(np.float16)
    rhs = np.random.randn(2, 2, 3).astype(np.float16)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1,),
        "padding": "SAME",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float16,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D convolution, complex64
    lhs = (np.random.randn(2, 2, 8, 8) + 1j * np.random.randn(2, 2, 8, 8)).astype(np.complex64)
    rhs = (np.random.randn(4, 2, 3, 3) + 1j * np.random.randn(4, 2, 3, 3)).astype(np.complex64)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1),
        "padding": "SAME",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.complex64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D convolution, negative float32 values
    lhs = -np.ones((2, 2, 4, 4), dtype=np.float32)
    rhs = np.ones((2, 2, 2, 2), dtype=np.float32)
    input_dict = {
        "lhs": lhs,
        "rhs": rhs,
        "window_strides": (1, 1),
        "padding": "VALID",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.float32,
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
