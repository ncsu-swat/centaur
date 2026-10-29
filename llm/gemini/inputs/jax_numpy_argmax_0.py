
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, keepdims=False
    a = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(0), "keepdims": False})
    )

    # Input 2: 2D array, axis=1, keepdims=True
    a = np.array([[1, 5, 2], [7, 3, 9]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(1), "keepdims": True})
    )

    # Input 3: 2D array with negative axis, keepdims=False
    a = np.random.randn(4, 5).astype(np.float64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(-1), "keepdims": False})
    )

    # Input 4: 3D array, axis=0, keepdims=True
    a = np.random.randint(-10, 10, size=(2, 3, 4), dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(0), "keepdims": True})
    )

    # Input 5: 3D array, axis=2, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(2), "keepdims": False})
    )

    # Input 6: 4D array, axis=-2, keepdims=True
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(-2), "keepdims": True})
    )

    # Input 7: Boolean array, axis=0, keepdims=False
    a = np.array([[False, True], [True, False]], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(0), "keepdims": False})
    )

    # Input 8: Array with duplicates to test tie-breaking (first occurrence), keepdims=True
    a = np.array([2, 3, 3, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(0), "keepdims": True})
    )

    # Input 9: Array with negative numbers, axis=1, keepdims=False
    a = np.array([[-5.0, -2.0, -10.0], [-1.0, -8.0, -3.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(1), "keepdims": False})
    )

    # Input 10: 5D array, axis=3, keepdims=True
    a = np.ones((2, 2, 2, 3, 2), dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(3), "keepdims": True})
    )

    # Input 11: 1D array with single element, axis=-1, keepdims=False
    a = np.array([42], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "axis": int(-1), "keepdims": False})
    )

    return list_of_inputs


generated_inputs["jax.numpy.argmax"] = argmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argmax'.")


check_valid('jax.numpy.argmax', generated_inputs['jax.numpy.argmax'], lib="jax", suffix=0)
