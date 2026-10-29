
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array with 1D indices
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    indices_or_sections = np.array([2, 4], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 2D array split into equal sections (0D scalar tensor)
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D array with 1D split points
    ary = np.random.randn(8, 3).astype(np.float64)
    indices_or_sections = np.array([2, 5], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D array with 1D split points
    ary = np.random.randint(-10, 10, size=(10, 4, 4), dtype=np.int32)
    indices_or_sections = np.array([3, 7], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 3D array split into equal sections (0D scalar tensor)
    ary = np.ones((9, 3, 2), dtype=np.int64)
    indices_or_sections = np.array(3, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 4D array with multiple split points
    ary = np.random.randn(12, 2, 3, 4).astype(np.float32)
    indices_or_sections = np.array([2, 4, 8, 10], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: Boolean array with single split point
    ary = np.array(
        [[True, False], [False, True], [True, True], [False, False]], dtype=bool
    )
    indices_or_sections = np.array([2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: Complex array with 0D section count
    ary = (np.random.randn(4, 5) + 1j * np.random.randn(4, 5)).astype(
        np.complex64
    )
    indices_or_sections = np.array(4, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: Empty split indices array (returns [ary])
    ary = np.random.randn(5, 5).astype(np.float32)
    indices_or_sections = np.array([], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 2D array with split point at boundary 0
    ary = np.arange(12, dtype=np.float32).reshape(4, 3)
    indices_or_sections = np.array([0, 2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 5D array split into equal sections
    ary = np.zeros((4, 2, 2, 2, 2), dtype=np.float32)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.vsplit_4"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_4'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_4'], lib="jax", suffix=4)
