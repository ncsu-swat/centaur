
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array split into 2 sections
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 1D float32 array split into 3 sections
    ary = np.arange(12, dtype=np.float32)
    indices_or_sections = 3
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D float64 array split into 2 sections
    ary = np.random.randn(4, 6).astype(np.float64)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 2D int64 array split into 4 sections
    ary = np.arange(24, dtype=np.int64).reshape(3, 8)
    indices_or_sections = 4
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 2D boolean array split into 1 section
    ary = np.array([[True, False, True, False], [False, True, False, True]])
    indices_or_sections = 1
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D float32 array split into 3 sections along axis 1
    ary = np.random.randn(2, 6, 4).astype(np.float32)
    indices_or_sections = 3
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D int16 array split into 5 sections along axis 1
    ary = np.ones((5, 10, 2), dtype=np.int16)
    indices_or_sections = 5
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 4D float32 array split into 2 sections along axis 1
    ary = np.random.randn(2, 4, 3, 2).astype(np.float32)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 1D complex64 array split into 4 sections
    ary = (np.arange(8) + 1j * np.arange(8)).astype(np.complex64)
    indices_or_sections = 4
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 2D uint8 array with single row split into 6 sections
    ary = np.arange(12, dtype=np.uint8).reshape(1, 12)
    indices_or_sections = 6
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 3D float64 array split into 4 sections along axis 1
    ary = np.zeros((3, 12, 5), dtype=np.float64)
    indices_or_sections = 4
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.hsplit_1"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_1'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_1'], lib="jax", suffix=1)
