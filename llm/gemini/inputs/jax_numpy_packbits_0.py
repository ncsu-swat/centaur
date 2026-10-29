
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def packbits_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, axis=0, bitorder='big'
    a = np.array([True, False, True, True, False, False, True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 0, "bitorder": "big"})
    )

    # Input 2: 1D int array with length not multiple of 8, axis=0, bitorder='little'
    a = np.array([1, 0, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 0, "bitorder": "little"})
    )

    # Input 3: 2D uint8 array, axis=1, bitorder='big'
    a = np.array(
        [[1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
         [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]],
        dtype=np.uint8
    )
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 1, "bitorder": "big"})
    )

    # Input 4: 2D array, packing along axis=0, bitorder='little'
    a = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 0, "bitorder": "little"})
    )

    # Input 5: 3D int32 array with negative axis, bitorder='big'
    a = np.random.choice([0, 1], size=(2, 4, 16)).astype(np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": -1, "bitorder": "big"})
    )

    # Input 6: 3D int64 array along middle axis, bitorder='little'
    a = np.random.choice([0, 1], size=(3, 9, 4)).astype(np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 1, "bitorder": "little"})
    )

    # Input 7: 4D boolean array, axis=2, bitorder='big'
    a = np.random.choice([True, False], size=(2, 3, 8, 2))
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 2, "bitorder": "big"})
    )

    # Input 8: 1D array of zeros, axis=-1, bitorder='little'
    a = np.zeros(24, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": -1, "bitorder": "little"})
    )

    # Input 9: 2D array with non-binary integer values, axis=0, bitorder='big'
    a = np.array([[5, -2, 0], [0, 3, 0], [1, 0, -1]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 0, "bitorder": "big"})
    )

    # Input 10: Empty 2D uint8 array, axis=1, bitorder='big'
    a = np.zeros((3, 0), dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": 1, "bitorder": "big"})
    )

    # Input 11: 4D int32 array, axis=-2, bitorder='little'
    a = np.random.randint(0, 2, size=(2, 2, 10, 3), dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": -2, "bitorder": "little"})
    )

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
