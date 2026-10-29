
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ravel_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, C-order
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 2: 2D float32 array, F-order
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 3: 1D int32 array, C-order
    a = np.array([10, -20, 30, -40, 50], dtype=np.int32)
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 4: 1D int32 array, F-order
    a = np.array([-1, 0, 1], dtype=np.int32)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 5: 3D int64 array, C-order
    a = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) - 10
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 6: 3D float64 array, F-order
    a = np.random.randn(3, 4, 2).astype(np.float64)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 7: 0D scalar array, C-order
    a = np.array(42.5, dtype=np.float32)
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 8: 4D float32 array with negative values, F-order
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 2)).astype(np.float32)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 9: 2D boolean array, C-order
    a = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 10: 2D complex64 array, F-order
    a = np.array([[1 + 2j, 3 - 4j], [-5 + 6j, 7 + 8j]], dtype=np.complex64)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 11: Empty 2D array, C-order
    a = np.empty((0, 5), dtype=np.float32)
    order = "C"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 12: 5D uint8 array, F-order
    a = np.zeros((1, 2, 1, 3, 2), dtype=np.uint8)
    order = "F"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    return list_of_inputs


generated_inputs["jax.numpy.ravel"] = ravel_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ravel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ravel'.")


check_valid('jax.numpy.ravel', generated_inputs['jax.numpy.ravel'], lib="jax", suffix=0)
