
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array split into 2 equal sections
    ary = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 2D array split into 4 equal sections
    ary = np.arange(16, dtype=np.int32).reshape(4, 4)
    indices_or_sections = 4
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D array split into 5 equal sections, float64
    ary = np.random.randn(10, 5).astype(np.float64)
    indices_or_sections = 5
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D array split into 3 sections
    ary = np.random.randn(6, 3, 2).astype(np.float32)
    indices_or_sections = 3
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 1D array with negative values, split into 4 sections
    ary = np.array([-8, -6, -4, -2, 0, 2, 4, 6], dtype=np.int64)
    indices_or_sections = 4
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 4D boolean array split into 2 sections
    ary = np.zeros((2, 2, 2, 2), dtype=bool)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 2D complex array split into 1 section
    ary = (np.random.randn(6, 2) + 1j * np.random.randn(6, 2)).astype(
        np.complex64
    )
    indices_or_sections = 1
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 3D uint8 array split into 6 sections
    ary = np.arange(12 * 4 * 4, dtype=np.uint8).reshape(12, 4, 4)
    indices_or_sections = 6
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 2D column vector split into 3 sections
    ary = np.linspace(-10.0, 10.0, 9, dtype=np.float32).reshape(9, 1)
    indices_or_sections = 3
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 5D array split into 2 sections
    ary = np.random.randn(4, 1, 2, 3, 1).astype(np.float32)
    indices_or_sections = 2
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: Single-element 1D array split into 1 section
    ary = np.array([42], dtype=np.int32)
    indices_or_sections = 1
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.vsplit_1"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_1'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_1'], lib="jax", suffix=1)
