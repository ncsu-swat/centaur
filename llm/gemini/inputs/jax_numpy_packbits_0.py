
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def packbits_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array, big-endian
    a = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    input_dict = {"a": a, "axis": 0, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple 1D array, little-endian
    a = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
    input_dict = {"a": a, "axis": 0, "bitorder": "little"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array with odd number of bits
    a = np.array([1, 0, 1, 1], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D Boolean array
    a = np.array([True, False, True, False, True, True, False, True], dtype=bool)
    input_dict = {"a": a, "axis": 0, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array packing along axis 1
    a = np.random.randint(0, 2, size=(3, 16)).astype(np.uint8)
    input_dict = {"a": a, "axis": 1, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array packing along axis 0
    a = np.random.randint(0, 2, size=(8, 4)).astype(np.uint8)
    input_dict = {"a": a, "axis": 0, "bitorder": "little"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array packing along axis 2
    a = np.random.randint(0, 2, size=(2, 3, 24)).astype(np.int8)
    input_dict = {"a": a, "axis": 2, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array packing along negative axis
    a = np.random.randint(0, 2, size=(2, 4, 8)).astype(np.uint8)
    input_dict = {"a": a, "axis": -1, "bitorder": "little"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array
    a = np.random.randint(0, 2, size=(100,)).astype(np.uint8)
    input_dict = {"a": a, "axis": 0, "bitorder": "big"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array packing along axis 2
    a = np.random.randint(0, 2, size=(2, 2, 8, 2)).astype(np.uint8)
    input_dict = {"a": a, "axis": 2, "bitorder": "little"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.packbits"] = packbits_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.packbits' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.packbits'.")


check_valid('jax.numpy.packbits', generated_inputs['jax.numpy.packbits'], lib="jax", suffix=0)
