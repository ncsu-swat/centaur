
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def compress_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0
    input_dict = {
        "condition": np.array([True, False, True, True]),
        "a": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        "axis": 0,
        "size": 3,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, compress along rows (axis=0)
    input_dict = {
        "condition": np.array([True, False, True]),
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
        "axis": 0,
        "size": 2,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, compress along columns (axis=1) with padding
    input_dict = {
        "condition": np.array([1, 0, 0, 1], dtype=np.int32),
        "a": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32),
        "axis": 1,
        "size": 4,
        "fill_value": 99.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, compress along axis=2
    input_dict = {
        "condition": np.array([True, True, False, True, False]),
        "a": np.random.randn(2, 3, 5).astype(np.float64),
        "axis": 2,
        "size": 5,
        "fill_value": 1.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative axis (axis=-1)
    input_dict = {
        "condition": np.array([False, True, True, False]),
        "a": np.ones((4, 2, 4), dtype=np.float32),
        "axis": -1,
        "size": 2,
        "fill_value": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Condition size matching axis length, size <= axis length
    input_dict = {
        "condition": np.array([True, False, True, False]),
        "a": np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int64),
        "axis": 1,
        "size": 2,
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, axis=1
    input_dict = {
        "condition": np.array([True, False, True]),
        "a": np.random.uniform(-5.0, 5.0, size=(2, 3, 2, 2)).astype(np.float32),
        "axis": 1,
        "size": 2,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All False condition with padding
    input_dict = {
        "condition": np.array([False, False, False, False]),
        "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "axis": 0,
        "size": 4,
        "fill_value": -0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: All True condition, axis=-2
    input_dict = {
        "condition": np.array([True, True, True]),
        "a": np.arange(9, dtype=np.float32).reshape(3, 3),
        "axis": -2,
        "size": 3,
        "fill_value": 10.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Size is smaller than number of matched elements (truncation)
    input_dict = {
        "condition": np.array([True, True, True, True]),
        "a": np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
        ),
        "axis": 0,
        "size": 1,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array, axis=0
    input_dict = {
        "condition": np.array([True, False, True]),
        "a": np.arange(24, dtype=np.float32).reshape(3, 2, 4),
        "axis": 0,
        "size": 2,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.compress_2"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_2'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_2'], lib="jax", suffix=2)
