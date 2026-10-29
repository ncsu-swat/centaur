
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 1D tensors, axes=0 (outer product), precision='default', preferred_element_type float32
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensors, axes=1 (vector dot product), precision='high', preferred_element_type float64
    a = np.array([1.5, -2.5, 3.5, 4.0], dtype=np.float64)
    b = np.array([-1.0, 0.5, 2.0, -3.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensors, axes=1 (matrix multiplication), precision='highest', preferred_element_type float32
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensors, axes=2 (matrix trace / inner product), precision='default', preferred_element_type float64
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors, axes=2, precision='high', preferred_element_type complex64
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    b = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensors, axes=1, precision='highest', preferred_element_type float32
    a = np.random.randn(5, 6, 7).astype(np.float32)
    b = np.random.randn(7, 2, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D tensors, axes=3, precision='default', preferred_element_type float32
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 3,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer arrays, axes=1, precision='default', preferred_element_type int32
    a = np.random.randint(-5, 5, size=(4, 3), dtype=np.int32)
    b = np.random.randint(-5, 5, size=(3, 2), dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 1,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D tensors, axes=2, precision='high', preferred_element_type float64
    a = np.random.randn(2, 4, 3, 5).astype(np.float64)
    b = np.random.randn(3, 5, 2, 6).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Higher-dimensional (5D and 3D), axes=2, precision='highest', preferred_element_type float32
    a = np.random.randn(1, 2, 2, 3, 4).astype(np.float32)
    b = np.random.randn(3, 4, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axes": 2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.tensordot_1"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_1'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_1'], lib="jax", suffix=1)
