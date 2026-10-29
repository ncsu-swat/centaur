
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array split into equal sections (0D tensor)
    ary = np.arange(12, dtype=np.float32)
    indices_or_sections = np.array(3, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 1D array split at specific 1D indices
    ary = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    indices_or_sections = np.array([2, 4], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D array split into equal sections (0D tensor)
    ary = np.arange(16, dtype=np.float64).reshape(4, 4)
    indices_or_sections = np.array(2, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 2D array split at specific 1D indices
    ary = np.random.randn(3, 8).astype(np.float32)
    indices_or_sections = np.array([1, 4, 6], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 3D array split into equal sections along horizontal axis (axis=1)
    ary = np.random.randn(2, 6, 4).astype(np.float32)
    indices_or_sections = np.array(3, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D array split at specific 1D indices
    ary = np.ones((4, 5, 2), dtype=np.int32)
    indices_or_sections = np.array([2, 3], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 4D array split with 1D index array
    ary = np.random.randn(2, 6, 2, 2).astype(np.float64)
    indices_or_sections = np.array([3], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 2D boolean array split into 1 section
    ary = np.array(
        [[True, False, True], [False, True, False]], dtype=np.bool_
    )
    indices_or_sections = np.array(1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 2D complex array split at multiple indices
    ary = (np.random.randn(3, 10) + 1j * np.random.randn(3, 10)).astype(
        np.complex64
    )
    indices_or_sections = np.array([2, 5, 7, 9], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 1D array split with an empty 1D indices array
    ary = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
    indices_or_sections = np.array([], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 2D array with negative numbers split into sections
    ary = np.linspace(-10, 10, 24).reshape(4, 6).astype(np.float32)
    indices_or_sections = np.array(6, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.hsplit_4"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_4'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_4'], lib="jax", suffix=4)
