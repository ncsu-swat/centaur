
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def swapaxes_inputs():
    list_of_inputs = []

    # Input 1: 2D array, swap axis 0 and 1 (standard transpose)
    a = np.ones((2, 3), dtype=np.float32)
    axis1 = 0
    axis2 = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 2: 3D array, swap axis 0 and 2
    a = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    axis1 = 0
    axis2 = 2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 3: 4D array with negative axis indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float64)
    axis1 = -1
    axis2 = -3
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 4: Swapping an axis with itself (identity permutation)
    a = np.zeros((3, 5, 7), dtype=np.complex64)
    axis1 = 1
    axis2 = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 5: 5D boolean array, mix of positive and negative axis
    a = np.random.choice([True, False], size=(2, 2, 3, 4, 2))
    axis1 = 1
    axis2 = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 6: 1D array, swapping axis 0 with 0
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    axis1 = 0
    axis2 = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 7: 1D array with negative axis
    a = np.array([10.0, 20.0, 30.0], dtype=np.float16)
    axis1 = -1
    axis2 = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 8: 6D array, swapping outer dimensions
    a = np.ones((1, 2, 1, 3, 1, 4), dtype=np.uint8)
    axis1 = 0
    axis2 = 5
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 9: 3D array with shape including dimension size 1
    a = np.random.randn(1, 10, 1).astype(np.float32)
    axis1 = 1
    axis2 = 2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 10: 4D array with both negative axes
    a = np.arange(120, dtype=np.int16).reshape((2, 3, 4, 5))
    axis1 = -4
    axis2 = -2
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    # Input 11: 2D array with negative axis swapped with positive axis
    a = np.eye(4, dtype=np.float32)
    axis1 = -2
    axis2 = 1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis1": axis1, "axis2": axis2})
    )

    return list_of_inputs


generated_inputs["jax.numpy.swapaxes"] = swapaxes_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.swapaxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.swapaxes'.")


check_valid('jax.numpy.swapaxes', generated_inputs['jax.numpy.swapaxes'], lib="jax", suffix=0)
