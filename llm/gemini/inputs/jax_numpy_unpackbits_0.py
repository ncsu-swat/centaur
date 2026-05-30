
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unpackbits_inputs():
    list_of_inputs = []
    
    # Input 1: 1D array, unpacking all bits, big-endian
    a = np.array([27], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 8,
        "bitorder": "big"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, partial unpacking, little-endian
    a = np.array([27], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 4,
        "bitorder": "little"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, negative count to trim bits, big-endian
    a = np.array([27, 154], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -1,
        "count": -2,
        "bitorder": "big"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, unpacking along axis 1, big-endian
    a = np.array([[154, 49]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "count": 16,
        "bitorder": "big"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, partial unpacking along axis 0, little-endian
    a = np.array([[154], [49]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 8,
        "bitorder": "little"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, partial unpacking along negative axis, big-endian
    a = np.array([[154, 49], [12, 34]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": -1,
        "count": 12,
        "bitorder": "big"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, full unpacking along axis 2, little-endian
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 2,
        "count": 24,
        "bitorder": "little"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, full unpacking along axis 0, big-endian
    a = np.ones((1, 2, 2), dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": 8,
        "bitorder": "big"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array, negative count to trim bits, little-endian
    a = np.array([255, 128], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 0,
        "count": -3,
        "bitorder": "little"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, negative count to trim bits along axis 1, big-endian
    a = np.array([[255, 128]], dtype=np.uint8)
    input_dict = {
        "a": a,
        "axis": 1,
        "count": -4,
        "bitorder": "big"
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
