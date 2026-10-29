
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rollaxis_inputs():
    list_of_inputs = []

    # Input 1: 1D array, rolling the only axis to 0
    a = np.arange(5, dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, rolling axis 1 to start 0
    a = np.ones((3, 4), dtype=np.int32)
    input_dict = {"a": a, "axis": 1, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, rolling axis 0 to end (start=2)
    a = np.zeros((2, 5), dtype=np.float64)
    input_dict = {"a": a, "axis": 0, "start": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, rolling axis 2 to start 0
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "axis": 2, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, rolling axis 1 to end (start=3)
    a = np.random.randint(0, 2, size=(3, 4, 5)).astype(bool)
    input_dict = {"a": a, "axis": 1, "start": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, negative axis
    a = (np.random.randn(4, 5, 6) + 1j * np.random.randn(4, 5, 6)).astype(
        np.complex64
    )
    input_dict = {"a": a, "axis": -1, "start": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, negative start
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 2, 2))
    input_dict = {"a": a, "axis": 2, "start": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, both negative axis and start
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "axis": -3, "start": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, rolling axis 3 to start 1
    a = np.ones((2, 2, 3, 4, 2), dtype=np.uint8)
    input_dict = {"a": a, "axis": 3, "start": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, rolling axis 0 to end (start=5)
    a = np.random.randn(3, 2, 4, 1, 2).astype(np.float32)
    input_dict = {"a": a, "axis": 0, "start": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array, axis == start (no-op)
    a = np.random.randn(4, 4, 4).astype(np.float64)
    input_dict = {"a": a, "axis": 1, "start": 1}
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
