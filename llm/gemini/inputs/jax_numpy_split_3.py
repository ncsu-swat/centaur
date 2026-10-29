
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array split at a single index
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = (5,)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 1D float array split at multiple indices
    ary = np.random.randn(12).astype(np.float32)
    indices_or_sections = (2, 6, 9)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D array split along axis 0
    ary = np.ones((6, 4), dtype=np.int32)
    indices_or_sections = (1, 3)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 2D float64 array split along axis 1
    ary = np.zeros((3, 8), dtype=np.float64)
    indices_or_sections = (2, 5, 7)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 2D array split along negative axis (-1)
    ary = np.arange(20, dtype=np.int64).reshape(4, 5)
    indices_or_sections = (1, 4)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 3D array split along axis 0
    ary = np.random.uniform(-1.0, 1.0, (4, 3, 2)).astype(np.float32)
    indices_or_sections = (2,)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 3D array split along axis 1
    ary = np.random.randn(2, 6, 3).astype(np.float64)
    indices_or_sections = (1, 3, 5)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 3D array split along axis 2
    ary = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    indices_or_sections = (1, 2, 3)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: 4D boolean array split along negative axis (-2)
    ary = np.ones((2, 3, 4, 5), dtype=np.bool_)
    indices_or_sections = (1, 2)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: Empty tuple of indices (returns single element list)
    ary = np.arange(5, dtype=np.int64)
    indices_or_sections = ()
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: Complex array split along axis 0
    ary = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j], dtype=np.complex64)
    indices_or_sections = (2,)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 12: Boundary indices (splits at 0 and length)
    ary = np.arange(8, dtype=np.float32)
    indices_or_sections = (0, 8)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.split_3"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_3'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_3'], lib="jax", suffix=3)
