
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def extract_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, matching 1D boolean mask
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    condition = arr % 2 == 0
    size = 6
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 2: 2D float32 array, matching 2D boolean mask
    arr = np.array([[-1.5, 2.3, 0.0], [4.1, -5.2, 6.7]], dtype=np.float32)
    condition = arr > 0
    size = 5
    fill_value = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 3: 1D int64 array with integer condition (evaluates truthiness)
    arr = np.array([-10, 20, -30, 40, 50], dtype=np.int64)
    condition = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    size = 4
    fill_value = np.array(999, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 4: 3D float64 array, 1D condition (mismatched size, condition smaller)
    arr = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    condition = np.array([True, False, True, True], dtype=bool)
    size = 3
    fill_value = np.array(np.nan, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 5: 1D uint8 array, condition larger than array (truncation)
    arr = np.array([10, 20, 30], dtype=np.uint8)
    condition = np.array(
        [True, False, True, True, False, True, False], dtype=bool
    )
    size = 2
    fill_value = np.array(255, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 6: 2D complex64 array with complex condition checking real part
    arr = np.array([[1 + 2j, -2 + 0j], [3 - 1j, -4 - 3j]], dtype=np.complex64)
    condition = arr.real > 0
    size = 4
    fill_value = np.array(0.0 + 0.0j, dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 7: 1D float32 array, all condition elements False
    arr = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    condition = np.zeros(arr.shape, dtype=bool)
    size = 4
    fill_value = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 8: 1D int16 array, all condition elements True
    arr = np.array([100, 200, 300], dtype=np.int16)
    condition = np.ones(arr.shape, dtype=bool)
    size = 3
    fill_value = np.array(-1, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 9: 4D int32 array, size smaller than extracted count
    arr = np.arange(16, dtype=np.int32).reshape((2, 2, 2, 2))
    condition = arr >= 0
    size = 2
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 10: 1D float32 array with size=1
    arr = np.array([0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    condition = arr < 0
    size = 1
    fill_value = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "arr": arr,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.extract"] = extract_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.extract' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.extract'.")


check_valid('jax.numpy.extract', generated_inputs['jax.numpy.extract'], lib="jax", suffix=0)
