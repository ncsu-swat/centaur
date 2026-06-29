
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cross_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D vectors, float32, axis=-1
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D vectors with negative values, float64, axis=0
    x1 = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    x2 = np.array([1.5, -2.5, 3.5], dtype=np.float64)
    input_dict = {"x1": x1, "x2": x2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, shape (3, 5), axis=0, int32
    x1 = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    input_dict = {"x1": x1, "x2": x2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, shape (4, 3), axis=1, float32
    x1 = np.random.randn(4, 3).astype(np.float32)
    x2 = np.random.randn(4, 3).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, shape (2, 3, 4), axis=1, float64
    x1 = np.random.randn(2, 3, 4).astype(np.float64)
    x2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {"x1": x1, "x2": x2, "axis": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays with broadcasting, axis=2
    x1 = np.random.randn(2, 1, 3).astype(np.float32)
    x2 = np.random.randn(2, 4, 3).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D arrays, shape (2, 2, 3, 2), axis=2
    x1 = np.random.randn(2, 2, 3, 2).astype(np.float32)
    x2 = np.random.randn(2, 2, 3, 2).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, shape (3, 2, 2), axis=0
    x1 = np.random.randn(3, 2, 2).astype(np.float32)
    x2 = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D arrays, broadcasting: (1, 3) and (5, 3), axis=-1
    x1 = np.random.randn(1, 3).astype(np.float32)
    x2 = np.random.randn(5, 3).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D arrays with broadcasting, axis=3
    x1 = np.random.randn(1, 1, 1, 3, 1).astype(np.float32)
    x2 = np.random.randn(2, 2, 2, 3, 5).astype(np.float32)
    input_dict = {"x1": x1, "x2": x2, "axis": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.cross"] = cross_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.cross'.")


check_valid('jax.numpy.linalg.cross', generated_inputs['jax.numpy.linalg.cross'], lib="jax", suffix=0)
