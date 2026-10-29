
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with positive and zero values
    a = np.array([0, 1, 0, 3, 5, 0], dtype=np.int32)
    size = 4
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float array with negative and positive values
    a = np.array([[-1.5, 0.0, 2.3], [0.0, -4.1, 0.0]], dtype=np.float32)
    size = 5
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 3: 3D boolean array
    a = np.array(
        [[[True, False], [False, True]], [[False, False], [True, True]]],
        dtype=np.bool_,
    )
    size = 6
    fill_value = np.array(99, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 4: 1D float64 array with size smaller than number of nonzeros
    a = np.array([10.5, 0.0, -3.2, 8.4, 0.0, 7.1], dtype=np.float64)
    size = 2
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 5: 2D int64 array with all nonzero values
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    size = 6
    fill_value = np.array(0, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 6: 2D array with all zeros
    a = np.zeros((3, 4), dtype=np.int32)
    size = 3
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 7: 4D float32 tensor
    a = np.zeros((2, 2, 2, 2), dtype=np.float32)
    a[0, 1, 0, 1] = 1.0
    a[1, 0, 1, 0] = -2.5
    size = 4
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 8: Complex array
    a = np.array([0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 2.0j, 0.0 + 0.0j], dtype=np.complex64)
    size = 3
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D array with size 0
    a = np.array([1, 0, 2, 0, 3], dtype=np.int32)
    size = 0
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D int8 array with negative fill value
    a = np.array([-128, 0, 127, 0, -1], dtype=np.int8)
    size = 5
    fill_value = np.array(-10, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

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
