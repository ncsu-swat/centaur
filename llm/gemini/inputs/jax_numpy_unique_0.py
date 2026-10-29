
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, axis=0
    ar = np.array([3, 4, 1, 3, 1, -2, -2], dtype=np.int32)
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": False,
        "return_counts": True,
        "axis": 0,
        "equal_nan": True,
        "size": 6,
        "fill_value": np.array(0, dtype=np.int32),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array along axis=0
    ar = np.array(
        [[1.0, 2.0], [2.0, 3.0], [1.0, 2.0], [np.nan, 1.0], [np.nan, 1.0]],
        dtype=np.float32,
    )
    input_dict = {
        "ar": ar,
        "return_index": False,
        "return_inverse": True,
        "return_counts": True,
        "axis": 0,
        "equal_nan": True,
        "size": 4,
        "fill_value": np.array(-1.0, dtype=np.float32),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float array along axis=1
    ar = np.array(
        [[1.0, 2.0, 1.0, 3.0], [4.0, 5.0, 4.0, 6.0]], dtype=np.float64
    )
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": True,
        "return_counts": False,
        "axis": 1,
        "equal_nan": False,
        "size": 3,
        "fill_value": np.array(0.0, dtype=np.float64),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int array along axis=0
    ar = np.array(
        [
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            [[-1, 0], [1, 2]],
        ],
        dtype=np.int64,
    )
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
        "axis": 0,
        "equal_nan": True,
        "size": 3,
        "fill_value": np.array(-99, dtype=np.int64),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int array along axis=2
    ar = np.ones((2, 3, 4), dtype=np.int32)
    input_dict = {
        "ar": ar,
        "return_index": False,
        "return_inverse": False,
        "return_counts": False,
        "axis": 2,
        "equal_nan": True,
        "size": 2,
        "fill_value": np.array(0, dtype=np.int32),
        "sorted": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float array with negative axis
    ar = np.array([-5.5, 0.0, 2.3, -5.5, 4.1], dtype=np.float32)
    input_dict = {
        "ar": ar,
        "return_index": False,
        "return_inverse": True,
        "return_counts": False,
        "axis": -1,
        "equal_nan": True,
        "size": 5,
        "fill_value": np.array(-10.0, dtype=np.float32),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array along axis=0
    ar = np.array(
        [[True, False], [False, True], [True, False], [True, True]], dtype=bool
    )
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": False,
        "return_counts": True,
        "axis": 0,
        "equal_nan": True,
        "size": 3,
        "fill_value": np.array(False, dtype=bool),
        "sorted": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array along axis=-1
    ar = np.array(
        [
            [10, 20, 10, 30],
            [10, 20, 10, 30],
        ],
        dtype=np.int16,
    )
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
        "axis": -1,
        "equal_nan": False,
        "size": 3,
        "fill_value": np.array(0, dtype=np.int16),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of uint8 values
    ar = np.array([255, 0, 128, 255, 0, 64], dtype=np.uint8)
    input_dict = {
        "ar": ar,
        "return_index": False,
        "return_inverse": False,
        "return_counts": True,
        "axis": 0,
        "equal_nan": True,
        "size": 4,
        "fill_value": np.array(0, dtype=np.uint8),
        "sorted": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array along axis=1
    ar = np.zeros((2, 4, 3, 2), dtype=np.float32)
    ar[:, 1, :, :] = 1.0
    ar[:, 2, :, :] = 1.0
    input_dict = {
        "ar": ar,
        "return_index": True,
        "return_inverse": True,
        "return_counts": True,
        "axis": 1,
        "equal_nan": True,
        "size": 3,
        "fill_value": np.array(-1.0, dtype=np.float32),
        "sorted": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.unique"] = unique_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique'.")


check_valid('jax.numpy.unique', generated_inputs['jax.numpy.unique'], lib="jax", suffix=0)
