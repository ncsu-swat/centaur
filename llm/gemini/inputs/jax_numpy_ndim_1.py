
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ndim_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar array (float32)
    a = np.array(3.14, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: 1-D array (int32)
    a = np.arange(10, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: 2-D array with negative values (float64)
    a = np.array([[-1.0, 2.5], [-3.2, 4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: 3-D array (int64)
    a = np.zeros((2, 3, 4), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: 4-D array (bool)
    a = np.ones((2, 2, 3, 1), dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: 5-D array (complex64)
    a = np.ones((1, 2, 1, 3, 1), dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: Empty 1-D array
    a = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: Empty 3-D array with zero-sized dimension
    a = np.zeros((0, 2, 5), dtype=np.int16)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: 2-D array (uint8)
    a = np.full((5, 5), 255, dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: 6-D array (float16)
    a = np.zeros((1, 1, 1, 1, 1, 1), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.numpy.ndim_1"] = ndim_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ndim_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ndim_1'.")


check_valid('jax.numpy.ndim', generated_inputs['jax.numpy.ndim_1'], lib="jax", suffix=1)
