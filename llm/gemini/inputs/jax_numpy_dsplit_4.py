
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def dsplit_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 array, split into 2 sections at index 2
    ary = np.random.randn(2, 3, 4).astype(np.float32)
    indices_or_sections = np.array([2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 3D float64 array with negative numbers, split at multiple indices
    ary = np.linspace(-10.0, 10.0, 36).reshape(2, 3, 6).astype(np.float64)
    indices_or_sections = np.array([2, 4], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 4D int32 array with negative values, split at index 1
    ary = np.random.randint(-50, 50, size=(2, 2, 4, 3)).astype(np.int32)
    indices_or_sections = np.array([1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D int64 array, split into multiple parts
    ary = np.arange(60, dtype=np.int64).reshape(3, 2, 10)
    indices_or_sections = np.array([2, 5, 8], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 5D float32 array, split along depth axis
    ary = np.random.randn(1, 2, 8, 2, 2).astype(np.float32)
    indices_or_sections = np.array([3, 6], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D complex64 array
    ary = (np.random.randn(2, 2, 6) + 1j * np.random.randn(2, 2, 6)).astype(
        np.complex64
    )
    indices_or_sections = np.array([1, 4], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D boolean array, split at index 2
    ary = np.random.choice([True, False], size=(3, 1, 5))
    indices_or_sections = np.array([2], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 3D uint8 array, split into three parts
    ary = np.random.randint(0, 255, size=(4, 3, 9), dtype=np.uint8)
    indices_or_sections = np.array([3, 6], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 4D float16 array with negative values
    ary = np.random.randn(2, 4, 6, 2).astype(np.float16)
    indices_or_sections = np.array([1, 3, 5], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 3D float32 array with empty split index array (single section)
    ary = np.random.randn(2, 2, 3).astype(np.float32)
    indices_or_sections = np.array([], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.dsplit_4"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dsplit_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dsplit_4'.")


check_valid('jax.numpy.dsplit', generated_inputs['jax.numpy.dsplit_4'], lib="jax", suffix=4)
