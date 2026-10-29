
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float arrays, size smaller than union
    input_dict = {
        "ar1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "ar2": np.array([2.0, 3.0, 4.0], dtype=np.float32),
        "size": 3,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float arrays with negative numbers, size larger than union (padding)
    input_dict = {
        "ar1": np.array([-5.0, -2.0, 0.0], dtype=np.float64),
        "ar2": np.array([-3.0, 0.0, 4.0], dtype=np.float64),
        "size": 8,
        "fill_value": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multi-dimensional arrays (flattened implicitly by union1d)
    input_dict = {
        "ar1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "ar2": np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "size": 6,
        "fill_value": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty first array
    input_dict = {
        "ar1": np.array([], dtype=np.float32),
        "ar2": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "size": 5,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Both arrays empty, size zero
    input_dict = {
        "ar1": np.array([], dtype=np.float64),
        "ar2": np.array([], dtype=np.float64),
        "size": 0,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Disjoint arrays with float64
    input_dict = {
        "ar1": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "ar2": np.array([40.0, 50.0, 60.0], dtype=np.float64),
        "size": 6,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Identical arrays with size larger than unique elements
    input_dict = {
        "ar1": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "ar2": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "size": 5,
        "fill_value": float("inf"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {
        "ar1": np.ones((2, 2, 2), dtype=np.float32),
        "ar2": np.zeros((1, 2, 3), dtype=np.float32),
        "size": 4,
        "fill_value": -0.5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-trivial duplicate elements within inputs
    input_dict = {
        "ar1": np.array([7.0, 7.0, 7.0, 8.0, 9.0], dtype=np.float32),
        "ar2": np.array([8.0, 8.0, 10.0, 11.0], dtype=np.float32),
        "size": 10,
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Size equals 1 (truncation to single element)
    input_dict = {
        "ar1": np.array([-10.0, 0.0, 10.0], dtype=np.float64),
        "ar2": np.array([-5.0, 5.0, 15.0], dtype=np.float64),
        "size": 1,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.union1d_3"] = union1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.union1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.union1d_3'.")


check_valid('jax.numpy.union1d', generated_inputs['jax.numpy.union1d_3'], lib="jax", suffix=3)
