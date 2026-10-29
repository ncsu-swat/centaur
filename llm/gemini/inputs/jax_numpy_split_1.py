
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def split_inputs():
    list_of_inputs = []

    # Input 1: 1D array, split into 2 equal sections along axis 0
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = 2
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 2: 2D array, split into 3 sections along axis 0
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = 3
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 3: 2D array, split into 2 sections along axis 1
    ary = np.ones((4, 8), dtype=np.float64)
    indices_or_sections = 2
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 4: 2D array, split along negative axis (-1)
    ary = np.arange(12, dtype=np.int64).reshape(3, 4)
    indices_or_sections = 4
    axis = -1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 5: 3D array, split along axis 1 into 2 sections
    ary = np.random.uniform(-1.0, 1.0, size=(2, 4, 6)).astype(np.float32)
    indices_or_sections = 2
    axis = 1
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 6: 3D array, split along axis 2 into 3 sections
    ary = np.zeros((2, 3, 9), dtype=np.int16)
    indices_or_sections = 3
    axis = 2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 7: 4D array, split along axis 0 into 1 section (identity split)
    ary = np.random.randn(5, 2, 2, 2).astype(np.float32)
    indices_or_sections = 1
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 8: 4D array, split along negative axis (-2) into 2 sections
    ary = np.arange(32, dtype=np.uint8).reshape(2, 2, 4, 2)
    indices_or_sections = 2
    axis = -2
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 9: Boolean 2D array, split along axis 0 into 4 sections
    ary = np.array(
        [[True, False], [False, True], [True, True], [False, False]], dtype=bool
    )
    indices_or_sections = 4
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 10: Complex 1D array, split along axis 0 into 5 sections
    ary = np.array([1 + 2j, 2 + 3j, 3 + 4j, 4 + 5j, 5 + 6j], dtype=np.complex64)
    indices_or_sections = 5
    axis = 0
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    # Input 11: 5D array, split along axis 3 into 2 sections
    ary = np.random.randn(1, 2, 2, 4, 3).astype(np.float32)
    indices_or_sections = 2
    axis = 3
    list_of_inputs.append(
        copy.deepcopy(
            {"ary": ary, "indices_or_sections": indices_or_sections, "axis": axis}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.split_1"] = split_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.split_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.split_1'.")


check_valid('jax.numpy.split', generated_inputs['jax.numpy.split_1'], lib="jax", suffix=1)
