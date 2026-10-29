
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def expand_dims_inputs():
    list_of_inputs = []

    # Input 1: 1D array, expand axis 0
    a = np.array([1, 2, 3], dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: 1D array, expand axis 1 (trailing)
    a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: 1D array, negative axis -1
    a = np.array([True, False, True], dtype=np.bool_)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: 1D array, negative axis -2
    a = np.array([1.0, 2.0], dtype=np.float32)
    axis = -2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: 2D array, expand axis 0
    a = np.ones((2, 3), dtype=np.float64)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: 2D array, expand axis 1 (middle)
    a = np.zeros((4, 5), dtype=np.int64)
    axis = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: 2D array, expand axis 2 (trailing)
    a = np.arange(6, dtype=np.int32).reshape((2, 3))
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: 2D array, negative axis -1
    a = np.random.randn(3, 4).astype(np.float32)
    axis = -1
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: 3D array, expand axis 0
    a = np.ones((2, 3, 4), dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: 3D array, expand axis 2
    a = np.zeros((2, 2, 2), dtype=np.float32)
    axis = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 11: 3D array, expand axis -3
    a = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    axis = -3
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 12: 0D scalar array, expand axis 0
    a = np.array(42, dtype=np.int32)
    axis = 0
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 13: 4D array, expand axis 4
    a = np.ones((1, 2, 3, 4), dtype=np.float32)
    axis = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs


generated_inputs["jax.numpy.expand_dims_1"] = expand_dims_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expand_dims_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expand_dims_1'.")


check_valid('jax.numpy.expand_dims', generated_inputs['jax.numpy.expand_dims_1'], lib="jax", suffix=1)
