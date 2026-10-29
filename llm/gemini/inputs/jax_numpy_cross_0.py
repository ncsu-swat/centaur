
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cross_inputs():
    list_of_inputs = []

    # Input 1: 1D vectors of length 3, float32, axis=-1
    a = np.array([1.0, 2.0, -3.0], dtype=np.float32)
    b = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D vectors of length 2, int32, axis=0
    a = np.array([2, -3], dtype=np.int32)
    b = np.array([-4, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 0,
        "axisb": 0,
        "axisc": 0,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays along axis 1 (length 3), float64
    a = np.random.randn(4, 3).astype(np.float64)
    b = np.random.randn(4, 3).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 1,
        "axisb": 1,
        "axisc": 1,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays along axis 0 (length 2), int64
    a = np.random.randint(-10, 10, size=(2, 5)).astype(np.int64)
    b = np.random.randint(-10, 10, size=(2, 5)).astype(np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 0,
        "axisb": 0,
        "axisc": 0,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays along axis 1 (length 3), float32
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 1,
        "axisb": 1,
        "axisc": 1,
        "axis": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D arrays along axis -1 (length 3), complex64
    a = (np.random.randn(3, 4, 3) + 1j * np.random.randn(3, 4, 3)).astype(
        np.complex64
    )
    b = (np.random.randn(3, 4, 3) + 1j * np.random.randn(3, 4, 3)).astype(
        np.complex64
    )
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays along axis 0 (length 3), float32
    a = np.random.randn(3, 2, 2).astype(np.float32)
    b = np.random.randn(3, 2, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 0,
        "axisb": 0,
        "axisc": 0,
        "axis": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays along axis 2 (length 3), float64
    a = np.random.randn(2, 4, 3, 5).astype(np.float64)
    b = np.random.randn(2, 4, 3, 5).astype(np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 2,
        "axisb": 2,
        "axisc": 2,
        "axis": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Broadcasting compatible arrays along axis -1 (length 3), float32
    a = np.random.randn(1, 3).astype(np.float32)
    b = np.random.randn(5, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": -1,
        "axisb": -1,
        "axisc": -1,
        "axis": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D broadcastable arrays along axis 1 (length 2), float32
    a = np.random.randn(3, 2, 1, 4).astype(np.float32)
    b = np.random.randn(3, 2, 5, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "axisa": 1,
        "axisb": 1,
        "axisc": 1,
        "axis": 1,
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
