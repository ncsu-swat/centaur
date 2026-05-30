
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def intersect1d_inputs():
    list_of_inputs = []

    # Input 1: Basic integer arrays
    ar1 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    ar2 = np.array([3, 4, 5, 6, 7], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": False,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 arrays with unique values
    ar1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ar2 = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": True,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int64 arrays, negative values
    ar1 = np.array([-5, -4, -3, -2, -1, 0], dtype=np.int64)
    ar2 = np.array([-3, -2, -1, 0, 1, 2], dtype=np.int64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 arrays, different lengths
    ar1 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    ar2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": True,
        "return_indices": False,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multidimensional arrays (flattened automatically by the API)
    ar1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ar2 = np.array([[3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean arrays
    ar1 = np.array([True, False, True], dtype=bool)
    ar2 = np.array([False, False, True], dtype=bool)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": False,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Arrays with duplicate values
    ar1 = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    ar2 = np.array([2, 2, 3, 3, 4, 4], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large range integers
    ar1 = np.array([1000, 2000, 3000], dtype=np.int32)
    ar2 = np.array([3000, 4000, 5000], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": True,
        "return_indices": False,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty first array
    ar1 = np.array([], dtype=np.int32)
    ar2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive and negative floats with duplicates
    ar1 = np.array([-1.5, 0.0, 1.5, 3.0, 3.0], dtype=np.float32)
    ar2 = np.array([-3.0, -1.5, 0.0, 1.5, 1.5], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "assume_unique": False,
        "return_indices": True,
        "size": None,
        "fill_value": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.intersect1d"] = intersect1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.intersect1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.intersect1d'.")


check_valid('jax.numpy.intersect1d', generated_inputs['jax.numpy.intersect1d'], lib="jax", suffix=0)
