
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def clip_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers, positive range
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    input_dict = {"arr": arr, "min": 2, "max": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array of floats, negative to positive range
    arr = np.random.randn(3, 4).astype(np.float32) * 10
    input_dict = {"arr": arr, "min": -2, "max": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array of floats with wider integer bounds
    arr = np.random.uniform(-50, 50, size=(2, 3, 4)).astype(np.float64)
    input_dict = {"arr": arr, "min": -10, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D scalar array
    arr = np.array(42, dtype=np.int64)
    input_dict = {"arr": arr, "min": 0, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array of int64 with negative bounds
    arr = np.random.randint(-100, 100, size=(2, 2, 3, 2), dtype=np.int64)
    input_dict = {"arr": arr, "min": -25, "max": -5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: min equals max (collapsing range)
    arr = np.arange(10, dtype=np.int32)
    input_dict = {"arr": arr, "min": 4, "max": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: min greater than max (valid in jax/numpy, returns max)
    arr = np.array([-5, 0, 5, 10], dtype=np.int32)
    input_dict = {"arr": arr, "min": 8, "max": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array of unsigned integers uint8
    arr = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
    input_dict = {"arr": arr, "min": 20, "max": 180}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array of float16
    arr = np.linspace(-10, 10, 12, dtype=np.float16).reshape(3, 4)
    input_dict = {"arr": arr, "min": -1, "max": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array with large integers
    arr = np.random.randint(-1000, 1000, size=(1, 2, 2, 2, 2), dtype=np.int32)
    input_dict = {"arr": arr, "min": -500, "max": 500}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D empty array
    arr = np.array([], dtype=np.float32)
    input_dict = {"arr": arr, "min": 0, "max": 10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.clip_3"] = clip_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.clip_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.clip_3'.")


check_valid('jax.numpy.clip', generated_inputs['jax.numpy.clip_3'], lib="jax", suffix=3)
