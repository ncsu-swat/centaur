
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array split at index 2
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices_or_sections = (2,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 1D float32 array split at indices 2 and 5
    ary = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    indices_or_sections = (2, 5)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D float64 array split at index 3 along horizontal axis
    ary = np.arange(12, dtype=np.float64).reshape(2, 6)
    indices_or_sections = (3,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 2D int64 array split at multiple indices
    ary = np.ones((4, 8), dtype=np.int64)
    indices_or_sections = (1, 3, 6)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 2D boolean array with empty tuple (returns single subarray)
    ary = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    indices_or_sections = ()
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D float32 array split along axis 1
    ary = np.random.randn(2, 6, 4).astype(np.float32)
    indices_or_sections = (2, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D complex64 array split at index 1
    ary = (np.random.randn(3, 4, 2) + 1j * np.random.randn(3, 4, 2)).astype(
        np.complex64
    )
    indices_or_sections = (1,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 4D int16 array split at indices 2 and 5
    ary = np.zeros((2, 7, 3, 2), dtype=np.int16)
    indices_or_sections = (2, 5)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 2D array split with index 0 (leading empty slice)
    ary = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    indices_or_sections = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 2D uint8 array split with duplicate indices
    ary = np.arange(20, dtype=np.uint8).reshape(4, 5)
    indices_or_sections = (2, 2, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 1D array split with index equal to array length
    ary = np.array([-5, -3, -1, 1, 3, 5], dtype=np.int32)
    indices_or_sections = (6,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.hsplit_3"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_3'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_3'], lib="jax", suffix=3)
