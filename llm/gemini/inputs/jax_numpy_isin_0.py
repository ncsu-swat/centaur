
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isin_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers, auto method
    element = np.array([1, 2, 3, 4], dtype=np.int32)
    test_elements = np.array([3, 4, 5, 6], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': False,
        'method': 'auto'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: invert=True, sort method
    element = np.array([10, -5, 0, 3], dtype=np.int32)
    test_elements = np.array([-5, 0], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': True,
        'method': 'sort'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: assume_unique=True, compare_all method, float32 type
    element = np.array([0.5, 1.5, -2.5], dtype=np.float32)
    test_elements = np.array([-2.5, 0.5, 3.5], dtype=np.float32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': True,
        'invert': False,
        'method': 'compare_all'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D elements, binary_search method
    element = np.array([[1, 2], [3, 4]], dtype=np.int64)
    test_elements = np.array([2, 4, 6], dtype=np.int64)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': False,
        'method': 'binary_search'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D elements, negative values, auto method
    element = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    test_elements = np.array([-2, -4, -6, -8], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': True,
        'method': 'auto'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D elements, 2D test_elements, sort method
    element = np.array([1, 2, 3], dtype=np.int32)
    test_elements = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': False,
        'method': 'sort'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 inputs, assume_unique=True, invert=True, compare_all method
    element = np.array([1e-5, 2e-5, 3e-5], dtype=np.float64)
    test_elements = np.array([2e-5, 4e-5], dtype=np.float64)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': True,
        'invert': True,
        'method': 'compare_all'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty element array
    element = np.array([], dtype=np.int32)
    test_elements = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': False,
        'method': 'auto'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty test_elements array
    element = np.array([1, 2, 3], dtype=np.int32)
    test_elements = np.array([], dtype=np.int32)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': False,
        'invert': False,
        'method': 'binary_search'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array elements
    element = np.array([True, False, True], dtype=bool)
    test_elements = np.array([True], dtype=bool)
    input_dict = {
        'element': element,
        'test_elements': test_elements,
        'assume_unique': True,
        'invert': False,
        'method': 'auto'
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
