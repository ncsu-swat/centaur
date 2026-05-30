
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rollaxis_inputs():
    list_of_inputs = []

    # Input 1: 3D array, roll axis 2 to start (0)
    a = np.ones((2, 3, 4), dtype=np.float32)
    input_dict = {"a": a, "axis": 2, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 4D array, roll axis 1 to end (4)
    a = np.zeros((3, 4, 5, 6), dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "start": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, negative axis, start at 0
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "axis": -1, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D array, axis 2, negative start (-2)
    a = np.random.randn(5, 4, 3, 2).astype(np.float64)
    input_dict = {"a": a, "axis": 2, "start": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis -2, negative start -1
    a = np.arange(27).reshape((3, 3, 3)).astype(np.int64)
    input_dict = {"a": a, "axis": -2, "start": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, roll axis 0 to start 2 (end)
    a = np.ones((10, 20), dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "start": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 5D array, axis 4, start 1
    a = np.zeros((2, 2, 2, 2, 2), dtype=np.uint8)
    input_dict = {"a": a, "axis": 4, "start": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, axis 1, start 0
    a = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"a": a, "axis": 1, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis 3, start 1
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict = {"a": a, "axis": 3, "start": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array, axis 0, start 3
    a = np.ones((4, 4, 4), dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "start": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 6D array, axis 5, start 2
    a = np.zeros((1, 2, 3, 4, 5, 6), dtype=np.float32)
    input_dict = {"a": a, "axis": 5, "start": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.rollaxis"] = rollaxis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rollaxis' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rollaxis'.")


check_valid('jax.numpy.rollaxis', generated_inputs['jax.numpy.rollaxis'], lib="jax", suffix=0)
