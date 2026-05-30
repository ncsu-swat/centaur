
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def median_inputs():
    list_of_inputs = []

    # Input 1: 1D array, odd size, positive float32
    a = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, even size, integers with negative values
    a = np.array([[-10, 20, 5], [30, -40, 0]], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array, float64, negative axis
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": -1,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, float32, axis=0
    a = np.random.randn(5, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D array, keepdims=True
    a = np.random.randn(2, 2, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 2,
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array, int64, keepdims=True
    a = np.array([100, -200, 300, -400], dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": -1,
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with uniform values
    a = np.ones((3, 4, 5), dtype=np.float32) * 4.5
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, large size, float32
    a = np.random.randn(100, 200).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, small dimensions, float64
    a = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": 3,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, containing duplicates
    a = np.array([[1.5, 1.5, 2.5], [3.5, 3.5, 3.5]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "overwrite_input": False,
        "keepdims": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array with negative axis and keepdims
    a = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "overwrite_input": False,
        "keepdims": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.median_1"] = median_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.median_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.median_1'.")


check_valid('jax.numpy.median', generated_inputs['jax.numpy.median_1'], lib="jax", suffix=1)
