
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, split evenly
    ary = np.arange(10, dtype=np.float32)
    indices_or_sections = 2
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 1D array, int32, uneven split
    ary = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    indices_or_sections = 4
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D array, float64, split along axis 0
    ary = np.random.randn(6, 4).astype(np.float64)
    indices_or_sections = 3
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 2D array, int64, split along axis 1
    ary = np.random.randint(-10, 10, size=(5, 7), dtype=np.int64)
    indices_or_sections = 3
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 2D array, bool, split with negative axis
    ary = np.random.choice([True, False], size=(4, 6))
    indices_or_sections = 2
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 3D array, complex64, split along axis 2
    ary = (np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)).astype(
        np.complex64
    )
    indices_or_sections = 2
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 3D array, uint8, split with indices_or_sections larger than dimension size
    ary = np.zeros((2, 3, 4), dtype=np.uint8)
    indices_or_sections = 5
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 4D array, float32, split along negative axis -2
    ary = np.random.randn(2, 6, 8, 3).astype(np.float32)
    indices_or_sections = 4
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 5D array, float64, 1 section (no-op split)
    ary = np.ones((2, 2, 2, 2, 2), dtype=np.float64)
    indices_or_sections = 1
    axis = 3
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: 1D empty array, float32
    ary = np.array([], dtype=np.float32)
    indices_or_sections = 3
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.array_split_1"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_1'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_1'], lib="jax", suffix=1)
