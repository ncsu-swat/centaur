
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, positive float divisor
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = 3.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D float array with negative numbers, positive float divisor
    x1 = np.array([-5.5, -4.2, 0.0, 1.7, 5.9], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float array, negative float divisor
    x1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    x2 = -1.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D integer array, fractional float divisor
    x1 = np.arange(12, dtype=np.int64).reshape(3, 4)
    x2 = 0.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float array, large float divisor
    x1 = np.random.uniform(-100, 100, size=(2, 3, 4)).astype(np.float32)
    x2 = 10.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar array, float divisor
    x1 = np.array(42, dtype=np.int32)
    x2 = 5.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float array, small float divisor
    x1 = np.ones((2, 2, 2, 2), dtype=np.float32)
    x2 = 0.25
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D array with positive and negative integers, negative float divisor
    x1 = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    x2 = -4.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Large 2D float64 array, float divisor
    x1 = np.linspace(-50.0, 50.0, 25).reshape(5, 5).astype(np.float64)
    x2 = 7.2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 1D uint8 array, float divisor
    x1 = np.array([0, 50, 100, 150, 200, 255], dtype=np.uint8)
    x2 = 25.5
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 3D integer array with zeros and ones, float divisor
    x1 = np.zeros((3, 2, 2), dtype=np.int32)
    x2 = 1.0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_3"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_3'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_3'], lib="jax", suffix=3)
