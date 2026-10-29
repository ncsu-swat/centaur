
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array split into equal sections (0D tensor sections)
    ary = np.arange(12, dtype=np.float32)
    indices_or_sections = np.array(3, dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 2D float64 array split by indices along axis 0
    ary = np.random.randn(8, 4).astype(np.float64)
    indices_or_sections = np.array([2, 5], dtype=np.int64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D int32 array split into equal sections along axis 1
    ary = np.arange(24, dtype=np.int32).reshape(4, 6)
    indices_or_sections = np.array(2, dtype=np.int64)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 3D float32 array split by indices along axis 1
    ary = np.random.randn(2, 9, 4).astype(np.float32)
    indices_or_sections = np.array([2, 6, 7], dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 3D int64 array split into equal sections along negative axis (-1)
    ary = np.random.randint(-100, 100, size=(3, 4, 8), dtype=np.int64)
    indices_or_sections = np.array(4, dtype=np.int32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 1D bool array split by indices along axis 0
    ary = np.array([True, False, True, True, False, False, True], dtype=bool)
    indices_or_sections = np.array([2, 4], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 4D float32 array split into equal sections along axis -2
    ary = np.random.randn(2, 3, 6, 4).astype(np.float32)
    indices_or_sections = np.array(3, dtype=np.int64)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 2D complex64 array split by indices along axis 0
    ary = (np.random.randn(6, 3) + 1j * np.random.randn(6, 3)).astype(
        np.complex64
    )
    indices_or_sections = np.array([1, 4], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 3D uint8 array split by a single index along axis 2
    ary = np.random.randint(0, 255, size=(4, 5, 6), dtype=np.uint8)
    indices_or_sections = np.array([3], dtype=np.int32)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: 1D float32 array split by an empty 1D tensor of indices
    ary = np.random.randn(5).astype(np.float32)
    indices_or_sections = np.array([], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: 2D int16 array split into 1 section (identity split)
    ary = np.arange(10, dtype=np.int16).reshape(2, 5)
    indices_or_sections = np.array(1, dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 12: 4D float64 array split by indices along negative axis (-3)
    ary = np.random.randn(4, 5, 3, 2).astype(np.float64)
    indices_or_sections = np.array([1, 3], dtype=np.int64)
    axis = -3
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.split_4"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_4'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_4'], lib="jax", suffix=4)
