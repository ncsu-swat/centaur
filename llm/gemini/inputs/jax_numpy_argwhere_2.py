
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argwhere_inputs():
    list_of_inputs = []

    # Input 1: 1D int array
    a = np.array([0, 1, 0, 2, 3, 0], dtype=np.int32)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float array with negative and zero values
    a = np.array([[-1.5, 0.0, 2.3], [0.0, -0.1, 0.0]], dtype=np.float32)
    size = 4
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D boolean array
    a = np.array(
        [[True, False, True], [False, False, True], [True, True, False]],
        dtype=bool,
    )
    size = 10
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D int array
    a = np.array(
        [
            [[1, 0], [0, 2]],
            [[0, 0], [3, -4]],
        ],
        dtype=np.int64,
    )
    size = 6
    fill_value = 99
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 5: 1D float array with truncation (size < number of nonzero elements)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    size = 3
    fill_value = -999
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 6: 4D integer array
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    a[0, 1, 0, 1] = 5
    a[1, 0, 1, 0] = -3
    size = 4
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 7: 2D complex array
    a = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 2.0j, -1.0 - 1.0j]], dtype=np.complex64)
    size = 5
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 8: 3D array of all zeros
    a = np.zeros((3, 3, 3), dtype=np.float32)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 9: 2D array with size 0
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    size = 0
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D uint8 array
    a = np.array([0, 255, 0, 128, 0], dtype=np.uint8)
    size = 8
    fill_value = 255
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.argwhere_2"] = argwhere_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argwhere_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argwhere_2'.")


check_valid('jax.numpy.argwhere', generated_inputs['jax.numpy.argwhere_2'], lib="jax", suffix=2)
