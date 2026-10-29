
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def shape_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar array (float32)
    a = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 1-D array with negative values (int32)
    a = np.array([-5, -2, 0, 3, 7], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 2-D matrix (float64)
    a = np.random.randn(4, 5).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 3-D tensor (complex64)
    a = (np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)).astype(
        np.complex64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: 4-D tensor (uint8)
    a = np.random.randint(0, 256, size=(2, 8, 8, 3), dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: 5-D tensor (float16)
    a = np.zeros((1, 2, 3, 4, 5), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: Empty 1-D array
    a = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: Empty 2-D array with zero dimension
    a = np.empty((0, 10), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: Boolean array
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 1-D array with a single element
    a = np.array([42], dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 11: Non-contiguous / transposed array
    a = np.random.randn(10, 20).T
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.shape_1"] = shape_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.shape_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.shape_1'.")


check_valid('jax.numpy.shape', generated_inputs['jax.numpy.shape_1'], lib="jax", suffix=1)
