
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def swapaxes_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array swap, float32
    a = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"a": a, "axis1": 0, "axis2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D array swap, int32
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {"a": a, "axis1": 1, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D array swap, float64
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict = {"a": a, "axis1": 0, "axis2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axes, boolean
    a = np.random.choice([True, False], size=(5, 5))
    input_dict = {"a": a, "axis1": -1, "axis2": -2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mixed positive/negative axes, complex64
    a = (np.random.randn(10, 20, 30) + 1j * np.random.randn(10, 20, 30)).astype(
        np.complex64
    )
    input_dict = {"a": a, "axis1": -3, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Higher dimension (5D) array, uint8
    a = np.random.randint(0, 255, size=(2, 2, 2, 2, 2)).astype(np.uint8)
    input_dict = {"a": a, "axis1": 1, "axis2": 4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Swapping same axis (no-op), float16
    a = np.random.randn(3, 3).astype(np.float16)
    input_dict = {"a": a, "axis1": 0, "axis2": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array with distinct dimensions, int64
    a = np.random.randint(-100, 100, size=(4, 2, 3)).astype(np.int64)
    input_dict = {"a": a, "axis1": 0, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with singleton dimensions, float32
    a = np.random.randn(1, 10, 1).astype(np.float32)
    input_dict = {"a": a, "axis1": 0, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array with mixed index types, int16
    a = np.random.randint(-50, 50, size=(5, 4, 3, 2, 1)).astype(np.int16)
    input_dict = {"a": a, "axis1": -2, "axis2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.swapaxes"] = swapaxes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.swapaxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.swapaxes'.")


check_valid('jax.numpy.swapaxes', generated_inputs['jax.numpy.swapaxes'], lib="jax", suffix=0)
