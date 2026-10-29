
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def squeeze_inputs():
    list_of_inputs = []

    # Input 1: 1D array of shape (1,), axis=0
    a = np.array([42], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 2D array of shape (1, 5), axis=0
    a = np.random.randn(1, 5).astype(np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 2D array of shape (4, 1), axis=1
    a = np.random.randn(4, 1).astype(np.float64)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 2D array of shape (4, 1), negative axis=-1
    a = np.zeros((4, 1), dtype=np.int64)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 3D array of shape (1, 3, 4), negative axis=-3
    a = np.ones((1, 3, 4), dtype=np.bool_)
    axis = -3
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 3D array of shape (3, 1, 2), axis=1
    a = np.random.randn(3, 1, 2).astype(np.complex64)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 4D array of shape (2, 3, 1, 5), axis=2
    a = np.arange(30).reshape(2, 3, 1, 5).astype(np.int16)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 4D array of shape (2, 3, 5, 1), negative axis=-1
    a = np.random.uniform(size=(2, 3, 5, 1)).astype(np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 5D array of shape (1, 2, 1, 3, 1), axis=0
    a = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 5D array of shape (1, 2, 1, 3, 1), axis=2
    a = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 5D array of shape (1, 2, 1, 3, 1), negative axis=-1
    a = np.random.randn(1, 2, 1, 3, 1).astype(np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.squeeze_1"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.squeeze_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.squeeze_1'.")


check_valid('jax.numpy.squeeze', generated_inputs['jax.numpy.squeeze_1'], lib="jax", suffix=1)
