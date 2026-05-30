
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D array, size=5, fill_value=-1
    a = np.array([1, 0, 2, 0, 3], dtype=np.int32)
    size = 5
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, size=3, fill_value=0
    a = np.array([[0, 5, 0], [6, 0, 8]], dtype=np.int32)
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, size=10, fill_value=-99
    a = np.array([[[1, 0], [0, 1]], [[0, 0], [1, 1]]], dtype=np.int32)
    size = 10
    fill_value = np.array(-99, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array float32, size=2, fill_value=-1
    a = np.array([0.0, 1.5, 0.0, -2.3], dtype=np.float32)
    size = 2
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D array, size=10, fill_value=-1
    a = np.zeros(50, dtype=np.int32)
    a[::5] = 1
    size = 10
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: All zeros array, size=5, fill_value=999
    a = np.zeros((3, 3), dtype=np.int32)
    size = 5
    fill_value = np.array(999, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All non-zeros array (truncation case), size=2, fill_value=-1
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    size = 2
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean array, size=4, fill_value=-1
    a = np.array([True, False, True, True], dtype=bool)
    size = 4
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, size=8, fill_value=-1
    a = np.ones((2, 2, 2, 2), dtype=np.int32)
    size = 8
    fill_value = np.array(-1, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with single element, size=1, fill_value=0
    a = np.array([0], dtype=np.int32)
    size = 1
    fill_value = np.array(0, dtype=np.int32)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.flatnonzero_1"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_1'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_1'], lib="jax", suffix=1)
