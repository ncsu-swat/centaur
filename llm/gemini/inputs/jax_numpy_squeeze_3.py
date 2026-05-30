
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def squeeze_inputs():
    list_of_inputs = []

    # Input 1: Simple 3D array with length-1 at axis 0, float32
    a = np.random.randn(1, 2, 3).astype(np.float32)
    axis = [0]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 2: Length-1 at axis 1, int32
    a = np.random.randint(0, 10, size=(2, 1, 3)).astype(np.int32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 3: Multiple length-1 axes, float64
    a = np.random.randn(1, 1, 5).astype(np.float64)
    axis = [0, 1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 4: Squeezing only one of the length-1 axes, float32
    a = np.random.randn(4, 1, 1, 3).astype(np.float32)
    axis = [1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 5: Squeezing outer length-1 axes, bool type
    a = np.random.choice([True, False], size=(1, 10, 1))
    axis = [0, 2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 6: Squeezing all dimensions of a 4D unit-shape array, int64
    a = np.ones((1, 1, 1, 1), dtype=np.int64)
    axis = [0, 1, 2, 3]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 7: Float32 with axes 1 and 3 having length 1
    a = np.random.randn(3, 1, 4, 1).astype(np.float32)
    axis = [1, 3]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 8: Negative axis index, float32
    a = np.random.randn(1, 5).astype(np.float32)
    axis = [-2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 9: Negative axis index for 3D array, float32
    a = np.random.randn(4, 1, 2).astype(np.float32)
    axis = [-2]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    # Input 10: Multiple negative axes, float32
    a = np.random.randn(1, 1, 1).astype(np.float32)
    axis = [-3, -1]
    list_of_inputs.append(copy.deepcopy({"a": a, "axis": axis}))

    return list_of_inputs

generated_inputs["jax.numpy.squeeze_3"] = squeeze_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.squeeze_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.squeeze_3'.")


check_valid('jax.numpy.squeeze', generated_inputs['jax.numpy.squeeze_3'], lib="jax", suffix=3)
