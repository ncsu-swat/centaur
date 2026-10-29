
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vdot_inputs():
    list_of_inputs = []

    # Input 1: 1D real float32 arrays
    a = np.array([1.0, -2.0, 3.5, -4.2, 5.0], dtype=np.float32)
    b = np.array([0.5, 2.0, -1.0, 3.0, -0.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex64 arrays
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    b = np.array([2.0 - 1.0j, 1.0 + 1.0j, -2.0 - 3.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with same shape
    a = np.arange(-3.0, 3.0, dtype=np.float32).reshape(2, 3)
    b = np.ones((2, 3), dtype=np.float32) * 2.5
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D complex128 arrays
    a = (np.arange(4, dtype=np.float64) + 1j * np.arange(4, dtype=np.float64)).reshape(2, 2)
    b = (np.ones((2, 2), dtype=np.float64) * 2.0 - 1j * np.ones((2, 2), dtype=np.float64)).astype(np.complex128)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 arrays with accumulation into int64
    a = np.array([[[1, -2], [3, 4]], [[-5, 6], [7, -8]]], dtype=np.int32)
    b = np.array([[[2, 1], [-1, 3]], [[4, -2], [0, 5]]], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different shapes but same total element count (2D and 1D)
    a = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    b = np.linspace(0.0, 2.0, 8, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D and 2D arrays with same total element count
    a = np.ones((2, 3, 2), dtype=np.float64)
    b = np.full((4, 3), -1.5, dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single element 1D arrays
    a = np.array([-4.5], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float32 promoted to complex64 via preferred_element_type
    a = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    b = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D tensors flattened with same total size
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.vdot_1"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_1'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_1'], lib="jax", suffix=1)
