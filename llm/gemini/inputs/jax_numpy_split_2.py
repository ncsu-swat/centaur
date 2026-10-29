
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split at two points
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = [3, 7]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 2D array, split along rows (axis 0)
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = [2, 4]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D array, split along columns (axis 1)
    ary = np.random.randn(5, 8).astype(np.float64)
    indices_or_sections = [1, 4, 6]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 3D array, split along axis 2
    ary = np.ones((3, 4, 12), dtype=np.int64)
    indices_or_sections = [3, 6, 9]
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 3D array, negative axis
    ary = np.random.randn(4, 6, 8).astype(np.float32)
    indices_or_sections = [2]
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: Boolean array, empty list of indices
    ary = np.array([True, False, True, True, False], dtype=np.bool_)
    indices_or_sections = []
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: Complex array, 2D, negative axis
    ary = (np.random.randn(6, 6) + 1j * np.random.randn(6, 6)).astype(
        np.complex64
    )
    indices_or_sections = [2, 5]
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 4D array, split along axis 1
    ary = np.arange(2 * 8 * 3 * 4, dtype=np.uint8).reshape(2, 8, 3, 4)
    indices_or_sections = [2, 4, 7]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 1D array, single split index
    ary = np.linspace(0.0, 1.0, 15, dtype=np.float32)
    indices_or_sections = [5]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: 2D array, indices within bounds
    ary = np.zeros((4, 5), dtype=np.int16)
    indices_or_sections = [1, 3, 5]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: 4D array, negative axis -3
    ary = np.random.randn(5, 7, 3, 2).astype(np.float32)
    indices_or_sections = [1, 4]
    axis = -3
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.split_2"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_2'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_2'], lib="jax", suffix=2)
