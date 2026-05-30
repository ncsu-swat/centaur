
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def roll_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive shift, axis 0
    a = np.arange(10).astype(np.int32)
    shift = 3
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, negative shift, axis 1
    a = np.random.randn(4, 5).astype(np.float32)
    shift = -2
    axis = 1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, shift larger than dimension, axis 2
    a = np.random.randint(0, 100, size=(2, 3, 4)).astype(np.int64)
    shift = 6
    axis = 2
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, shift is zero, axis 0
    a = np.random.randn(3, 3).astype(np.float64)
    shift = 0
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, negative axis, shift 1
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    shift = 1
    axis = -1
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean array, negative shift, axis 0
    a = np.random.choice([True, False], size=(5, 2))
    shift = -1
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array, large negative shift, axis 0
    a = np.arange(5).astype(np.float32)
    shift = -12
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis 0, positive shift
    a = np.random.randn(5, 1, 5).astype(np.float32)
    shift = 4
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, large size, positive shift, axis -2
    a = np.random.randn(100, 10).astype(np.float32)
    shift = 50
    axis = -2
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with single element, shift 10, axis 0
    a = np.array([42]).astype(np.int32)
    shift = 10
    axis = 0
    input_dict = {"a": a, "shift": shift, "axis": axis}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.roll_1"] = roll_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.roll_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.roll_1'.")


check_valid('jax.numpy.roll', generated_inputs['jax.numpy.roll_1'], lib="jax", suffix=1)
