
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single split point
    ary = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    indices_or_sections = [3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 2: 1D array with negative values, multiple split points
    ary = np.array([-10.5, -2.1, 0.0, 3.4, 7.8, -5.2], dtype=np.float32)
    indices_or_sections = [2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 3: 2D array, split columns at multiple indices
    ary = np.arange(16, dtype=np.float64).reshape(4, 4)
    indices_or_sections = [1, 3]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 4: 2D array, empty list
    ary = np.ones((3, 5), dtype=np.int64)
    indices_or_sections = []
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 5: 2D boolean array
    ary = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    indices_or_sections = [1, 2]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 6: 3D array, splits along axis 1
    ary = np.random.randn(2, 6, 4).astype(np.float32)
    indices_or_sections = [2, 5]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 7: 3D array with complex numbers
    ary = (
        np.arange(24).reshape(2, 4, 3) + 1j * np.arange(24).reshape(2, 4, 3)
    ).astype(np.complex64)
    indices_or_sections = [1]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 8: 4D array
    ary = np.arange(48, dtype=np.int16).reshape(2, 6, 2, 2)
    indices_or_sections = [1, 3, 5]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 9: 2D array with split at boundary
    ary = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices_or_sections = [1, 2, 4]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 10: 1D array with uint8 data type
    ary = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
    indices_or_sections = [2, 4, 6]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    # Input 11: 2D array with 1 column
    ary = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    indices_or_sections = [0]
    list_of_inputs.append(
        copy.deepcopy({"ary": ary, "indices_or_sections": indices_or_sections})
    )

    return list_of_inputs


generated_inputs["jax.numpy.hsplit_2"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hsplit_2'.")


check_valid('jax.numpy.hsplit', generated_inputs['jax.numpy.hsplit_2'], lib="jax", suffix=2)
