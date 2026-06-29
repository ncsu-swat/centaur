
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def vecdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays, axis=-1
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays, axis=0
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 arrays, axis=1
    x1 = np.random.randint(-10, 10, size=(2, 5)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(2, 5)).astype(np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 1,
        "precision": "default",
        "preferred_element_type": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 arrays, axis=1, high precision
    x1 = np.random.randn(2, 4, 3).astype(np.float64)
    x2 = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 1,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 arrays, axis=-1
    x1 = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    x2 = (np.random.randn(4) + 1j * np.random.randn(4)).astype(np.complex64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "default",
        "preferred_element_type": np.complex64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D broadcasted arrays, axis=2
    x1 = np.random.randn(2, 1, 3, 4).astype(np.float32)
    x2 = np.random.randn(1, 5, 3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 2,
        "precision": "high",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float16 arrays, cast output to float32
    x1 = np.random.randn(8).astype(np.float16)
    x2 = np.random.randn(8).astype(np.float16)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "default",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays with negative values, axis=-1
    x1 = np.random.uniform(-5.0, 5.0, size=(2, 2, 6)).astype(np.float32)
    x2 = np.random.uniform(-5.0, 5.0, size=(2, 2, 6)).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": "highest",
        "preferred_element_type": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D int64 arrays, axis=0
    x1 = np.random.randint(-100, 100, size=(100,)).astype(np.int64)
    x2 = np.random.randint(-100, 100, size=(100,)).astype(np.int64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "default",
        "preferred_element_type": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 with broadcasted dimensions, axis=0
    x1 = np.random.randn(5, 1).astype(np.float64)
    x2 = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": "highest",
        "preferred_element_type": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.vecdot_1"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vecdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vecdot_1'.")


check_valid('jax.numpy.linalg.vecdot', generated_inputs['jax.numpy.linalg.vecdot_1'], lib="jax", suffix=1)
