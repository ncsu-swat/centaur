
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_vecdot_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 vectors, default axis, default precision
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D complex vectors, custom axis, high precision
    x1 = np.array([1j, 2j], dtype=np.complex64)
    x2 = np.array([3.0, 4.0], dtype=np.complex64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 arrays, axis -1, highest precision
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer arrays, axis 0, default precision
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D random float32 arrays, axis 1, high precision
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float64 arrays, axis 2, highest precision
    x1 = np.random.randn(2, 2, 3, 3).astype(np.float64)
    x2 = np.random.randn(2, 2, 3, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D vectors with negative values
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([-4.0, 5.0, -6.0], dtype=np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D arrays with broadcasting, axis -1, high precision
    x1 = np.random.randn(2, 3).astype(np.float32)
    x2 = np.random.randn(1, 3).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float64 arrays, axis -2, highest precision
    x1 = np.random.randn(2, 4, 3).astype(np.float64)
    x2 = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -2,
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D integer arrays returning a float32 accumulator type
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vecdot_1"] = jax_numpy_vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vecdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vecdot_1'.")


check_valid('jax.numpy.vecdot', generated_inputs['jax.numpy.vecdot_1'], lib="jax", suffix=1)
