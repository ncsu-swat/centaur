
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cross_inputs():
    list_of_inputs = []

    # Input 1: 1D vectors of size 3 (standard 3D cross product)
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D vectors of size 2 (2D cross product, returns scalar)
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batched 3D vectors (2D arrays), shape (5, 3)
    a = np.random.randn(5, 3).astype(np.float32)
    b = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched 2D vectors (2D arrays), axis=0, shape (2, 5)
    a = np.random.randn(2, 5).astype(np.float32)
    b = np.random.randn(2, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 0,
        "axisb": 0,
        "axisc": 0,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays, cross product along axis 1 (size 3)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 1,
        "axisb": 1,
        "axisc": 1,
        "axis": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes, a: (4, 3), b: (1, 3)
    a = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(1, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 inputs, 3D vectors
    a = np.random.randn(3).astype(np.float64)
    b = np.random.randn(3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer inputs, 2D vectors with negative values
    a = np.array([-1, 5], dtype=np.int32)
    b = np.array([3, -2], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 0,
        "axisb": 0,
        "axisc": 0,
        "axis": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large batch of 3D vectors, integer type
    a = np.random.randint(-10, 10, size=(10, 3)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(10, 3)).astype(np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D arrays, cross product along axis 2 (size 2)
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    b = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 2,
        "axisb": 2,
        "axisc": 2,
        "axis": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cross"] = cross_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cross'.")


check_valid('jax.numpy.cross', generated_inputs['jax.numpy.cross'], lib="jax", suffix=0)
