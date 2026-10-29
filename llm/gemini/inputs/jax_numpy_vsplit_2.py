
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with single split point
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices_or_sections = [3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 2D float32 array with single split point
    ary = np.random.randn(6, 4).astype(np.float32)
    indices_or_sections = [2]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D float64 array with multiple split points
    ary = np.random.randn(8, 5).astype(np.float64)
    indices_or_sections = [2, 5]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 3D int32 array with multiple split points
    ary = np.arange(60, dtype=np.int32).reshape(6, 2, 5)
    indices_or_sections = [1, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 4D float32 array
    ary = np.random.randn(4, 3, 2, 2).astype(np.float32)
    indices_or_sections = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 2D boolean array
    ary = np.array(
        [[True, False], [False, True], [True, True], [False, False]], dtype=bool
    )
    indices_or_sections = [1, 2, 3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 2D complex64 array
    ary = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(
        np.complex64
    )
    indices_or_sections = [2]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 1D float32 array with negative values
    ary = np.array([-5.5, -2.1, 0.0, 3.4, 7.8, -1.2], dtype=np.float32)
    indices_or_sections = [2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 3D float32 array with single split point
    ary = np.random.randn(5, 3, 3).astype(np.float32)
    indices_or_sections = [2]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 2D int64 array with multiple valid split points
    ary = np.arange(32, dtype=np.int64).reshape(8, 4)
    indices_or_sections = [2, 6]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 5D float32 array
    ary = np.random.randn(6, 2, 2, 2, 2).astype(np.float32)
    indices_or_sections = [2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.vsplit_2"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_2'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_2'], lib="jax", suffix=2)
