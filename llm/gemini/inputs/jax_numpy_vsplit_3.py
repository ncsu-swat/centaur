
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array with two split points
    ary = np.arange(10, dtype=np.int32)
    indices_or_sections = (3, 7)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 1D array with a single split point
    ary = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    indices_or_sections = (4,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D float64 array with two split points
    ary = np.random.randn(6, 4).astype(np.float64)
    indices_or_sections = (2, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 2D int64 array with multiple split points
    ary = np.random.randint(-10, 10, size=(8, 3), dtype=np.int64)
    indices_or_sections = (1, 3, 6)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 2D boolean array
    ary = np.array([[True, False], [False, True], [True, True], [False, False]])
    indices_or_sections = (2,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D float32 array
    ary = np.random.randn(6, 3, 2).astype(np.float32)
    indices_or_sections = (1, 3, 5)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D complex64 array
    ary = (np.random.randn(4, 2, 2) + 1j * np.random.randn(4, 2, 2)).astype(
        np.complex64
    )
    indices_or_sections = (2,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 4D float32 array
    ary = np.random.randn(5, 2, 3, 4).astype(np.float32)
    indices_or_sections = (1, 4)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 1D array with fine-grained splits
    ary = np.arange(8, dtype=np.float32)
    indices_or_sections = (1, 2, 3, 4, 5, 6, 7)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 2D int16 array with negative values
    ary = np.random.randint(-100, 100, size=(10, 5), dtype=np.int16)
    indices_or_sections = (5,)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 3D float64 array with multiple split points
    ary = np.random.randn(7, 4, 1).astype(np.float64)
    indices_or_sections = (2, 4, 6)
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.vsplit_3"] = vsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vsplit_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vsplit_3'.")


check_valid('jax.numpy.vsplit', generated_inputs['jax.numpy.vsplit_3'], lib="jax", suffix=3)
