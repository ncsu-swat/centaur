
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isin_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays with default-like settings
    element = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    test_elements = np.array([2, 4, 6], dtype=np.int32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "auto",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 arrays with invert=True
    element = np.array([[1.0, -2.5], [3.2, 4.0]], dtype=np.float32)
    test_elements = np.array([-2.5, 0.0, 4.0, 5.5], dtype=np.float32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": True,
        "method": "auto",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int64 arrays, assume_unique=True, method='compare_all'
    element = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    test_elements = np.array([-5, 5, 15], dtype=np.int64)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": False,
        "method": "compare_all",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array and 1D test elements with method='binary_search'
    element = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    test_elements = np.array([0, 5, 10, 15, 20], dtype=np.int32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "binary_search",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64 arrays with method='sort'
    element = np.array([-1.5, 0.0, 1.5, 3.0], dtype=np.float64)
    test_elements = np.array([[-1.5, 2.0], [3.0, 4.5]], dtype=np.float64)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": True,
        "method": "sort",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D uint8 arrays, invert=True, assume_unique=True
    element = np.array([10, 20, 30, 40], dtype=np.uint8)
    test_elements = np.array([20, 40, 60], dtype=np.uint8)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": True,
        "method": "auto",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D (scalar-like) element and 1D test elements
    element = np.array(7, dtype=np.int32)
    test_elements = np.array([1, 3, 5, 7, 9], dtype=np.int32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "compare_all",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boolean arrays
    element = np.array([True, False, True], dtype=bool)
    test_elements = np.array([False], dtype=bool)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "auto",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty element array
    element = np.array([], dtype=np.int32)
    test_elements = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "auto",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty test_elements array
    element = np.array([[1, 2], [3, 4]], dtype=np.int64)
    test_elements = np.array([], dtype=np.int64)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": False,
        "invert": False,
        "method": "sort",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D element and 2D test_elements with negative values
    element = np.array([[-8, -3], [0, 4]], dtype=np.int32)
    test_elements = np.array([[-8, 12], [4, -100]], dtype=np.int32)
    input_dict = {
        "element": element,
        "test_elements": test_elements,
        "assume_unique": True,
        "invert": False,
        "method": "binary_search",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.isin"] = isin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isin'.")


check_valid('jax.numpy.isin', generated_inputs['jax.numpy.isin'], lib="jax", suffix=0)
