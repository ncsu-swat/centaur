
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def resize_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int array resized to 2D square matrix
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    new_shape = (3, 3)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 2: 1D int array resized to a larger 2D matrix (repetition)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
    new_shape = (3, 4)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 3: 2D float32 array resized to a larger 2D matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    new_shape = (4, 4)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 4: 2D float64 array resized to a smaller 2D matrix
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    new_shape = (2, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 5: 3D int16 array resized to a 2D matrix
    a = np.arange(8, dtype=np.int16).reshape((2, 2, 2))
    new_shape = (4, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 6: 1D negative float array resized to a 2D matrix
    a = np.array([-1.5, -2.5, -3.5, -4.5, -5.5], dtype=np.float32)
    new_shape = (2, 3)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 7: Boolean array resized to a 2D matrix
    a = np.array([True, False, True], dtype=np.bool_)
    new_shape = (2, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 8: 3D float32 array flattened to a 1D array of different size
    a = np.ones((2, 3, 4), dtype=np.float32)
    new_shape = (12,)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 9: 1D complex array resized to 2D
    a = np.array([1+2j, 3+4j], dtype=np.complex64)
    new_shape = (2, 2)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    # Input 10: Larger 2D array resized to a different aspect ratio
    a = np.arange(100, dtype=np.int32).reshape((10, 10))
    new_shape = (5, 20)
    list_of_inputs.append({"a": copy.deepcopy(a), "new_shape": new_shape})

    return list_of_inputs

generated_inputs["jax.numpy.resize_2"] = resize_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.resize_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.resize_2'.")


check_valid('jax.numpy.resize', generated_inputs['jax.numpy.resize_2'], lib="jax", suffix=2)
