
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_split_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, split at index [3]
    ary = np.arange(10, dtype=np.float32)
    indices_or_sections = [3]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 1D int32 array, multiple split indices [2, 5, 7]
    ary = np.arange(12, dtype=np.int32)
    indices_or_sections = [2, 5, 7]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D float64 array, split along axis 0
    ary = np.ones((8, 6), dtype=np.float64)
    indices_or_sections = [2, 4, 6]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 2D int64 array, split along axis 1
    ary = np.arange(24, dtype=np.int64).reshape(4, 6)
    indices_or_sections = [1, 3]
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 2D boolean array, negative axis -1
    ary = np.zeros((5, 8), dtype=bool)
    indices_or_sections = [2, 5]
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 3D float32 array, split along axis 2
    ary = np.random.randn(3, 4, 10).astype(np.float32)
    indices_or_sections = [2, 6, 8]
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 3D complex64 array, negative axis -2
    ary = np.ones((2, 6, 4), dtype=np.complex64)
    indices_or_sections = [1, 4]
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 4D uint8 array, split along axis 0
    ary = np.ones((6, 2, 2, 2), dtype=np.uint8)
    indices_or_sections = [1, 3, 5]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 1D float16 array, empty list (returns [ary])
    ary = np.linspace(0, 1, 10, dtype=np.float16)
    indices_or_sections = []
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: 2D float32 array, split boundary at exact dimension length
    ary = np.random.randn(4, 5).astype(np.float32)
    indices_or_sections = [2, 4]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: 4D float64 array, split along axis 3
    ary = np.random.randn(2, 3, 4, 8).astype(np.float64)
    indices_or_sections = [2, 4, 6]
    axis = 3
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 12: 1D int16 array, split at index 0
    ary = np.array([10, 20, 30], dtype=np.int16)
    indices_or_sections = [0]
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.array_split_2"] = array_split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_split_2'.")


check_valid('jax.numpy.array_split', generated_inputs['jax.numpy.array_split_2'], lib="jax", suffix=2)
