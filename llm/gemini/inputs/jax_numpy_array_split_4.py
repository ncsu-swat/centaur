
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, split using 1D index array, axis 0
    ary = np.arange(10, dtype=np.float32)
    indices_or_sections = np.array([3, 7], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 2D int32 array, split along axis 0
    ary = np.arange(12, dtype=np.int32).reshape(6, 2)
    indices_or_sections = np.array([2, 4], dtype=np.int64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D float64 array, split along axis 1
    ary = np.random.randn(4, 8).astype(np.float64)
    indices_or_sections = np.array([2, 5, 7], dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 2D float32 array, split along negative axis -1
    ary = np.ones((5, 10), dtype=np.float32)
    indices_or_sections = np.array([3, 6], dtype=np.int32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 3D float32 array, split along axis 1
    ary = np.random.randn(3, 9, 4).astype(np.float32)
    indices_or_sections = np.array([3, 6], dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 3D complex64 array, split along axis 2
    ary = (np.random.randn(2, 3, 6) + 1j * np.random.randn(2, 3, 6)).astype(
        np.complex64
    )
    indices_or_sections = np.array([1, 4], dtype=np.int32)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 4D int64 array, split along axis -2
    ary = np.arange(48, dtype=np.int64).reshape(2, 3, 4, 2)
    indices_or_sections = np.array([1, 3], dtype=np.int32)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 1D boolean array, split at a single index
    ary = np.array([True, False, True, True, False, False], dtype=np.bool_)
    indices_or_sections = np.array([4], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 2D array, split with empty index array
    ary = np.random.randn(5, 5).astype(np.float32)
    indices_or_sections = np.array([], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: 1D float32 array, split within bounds
    ary = np.arange(8, dtype=np.float32)
    indices_or_sections = np.array([1, 4, 7], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: 3D float32 array, split along axis 0
    ary = np.random.uniform(-5.0, 5.0, size=(8, 2, 2)).astype(np.float32)
    indices_or_sections = np.array([1, 3, 6], dtype=np.int64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.array_split_4"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_4'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_4'], lib="jax", suffix=4)
