
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unpackbits_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, count positive, bitorder 'big'
    a = np.array([2, 7, 23], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 16,
        "bitorder": "big",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, axis=-1, count negative, bitorder 'little'
    a = np.array([128, 64, 32, 16], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -1,
        "count": -5,
        "bitorder": "little",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=1, count positive, bitorder 'big'
    a = np.array([[154, 49], [12, 200]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "count": 11,
        "bitorder": "big",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, axis=0, count negative, bitorder 'little'
    a = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": -8,
        "bitorder": "little",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, axis=2, count positive, bitorder 'little'
    a = np.random.randint(0, 256, size=(2, 3, 4), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 2,
        "count": 25,
        "bitorder": "little",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=-2, count negative, bitorder 'big'
    a = np.random.randint(0, 256, size=(4, 2, 3), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -2,
        "count": -4,
        "bitorder": "big",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, axis=1, count positive, bitorder 'big'
    a = np.random.randint(0, 256, size=(2, 2, 2, 2), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "count": 12,
        "bitorder": "big",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with single element, axis=0, exact count, bitorder 'little'
    a = np.array([255], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 8,
        "bitorder": "little",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, axis=-1, count=1, bitorder 'big'
    a = np.zeros((3, 3), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -1,
        "count": 1,
        "bitorder": "big",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, axis=3, count negative, bitorder 'little'
    a = np.ones((1, 2, 1, 3), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 3,
        "count": -10,
        "bitorder": "little",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unpackbits"] = unpackbits_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unpackbits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unpackbits'.")


check_valid('jax.numpy.unpackbits', generated_inputs['jax.numpy.unpackbits'], lib="jax", suffix=0)
