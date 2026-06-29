
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_inputs():
    list_of_inputs = []

    # Input 1: Basic positive scalars
    input_dict = {
        "x": int(2),
        "loc": int(1),
        "scale": int(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zero and negatives
    input_dict = {
        "x": int(-5),
        "loc": int(-2),
        "scale": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Numpy int32 scalars
    input_dict = {
        "x": np.int32(10),
        "loc": np.int32(0),
        "scale": np.int32(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Numpy int64 scalars
    input_dict = {
        "x": np.int64(-1),
        "loc": np.int64(3),
        "scale": np.int64(10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D numpy array of integers
    input_dict = {
        "x": np.array([1, 2, 3], dtype=np.int32),
        "loc": np.array([0, 1, 2], dtype=np.int32),
        "scale": np.array([1, 2, 3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D numpy array of integers
    input_dict = {
        "x": np.array([[1, -2], [3, -4]], dtype=np.int64),
        "loc": np.array([[0, 1], [-1, 2]], dtype=np.int64),
        "scale": np.array([[2, 3], [1, 4]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting scalar loc/scale with 1D array x
    input_dict = {
        "x": np.array([-10, 0, 10], dtype=np.int32),
        "loc": int(0),
        "scale": int(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting 1D array with scalar scale and 2D x
    input_dict = {
        "x": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "loc": np.array([1, 2], dtype=np.int32),
        "scale": int(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D numpy array of integers
    input_dict = {
        "x": np.ones((2, 2, 2), dtype=np.int32) * 5,
        "loc": np.zeros((2, 2, 2), dtype=np.int32),
        "scale": np.ones((2, 2, 2), dtype=np.int32) * 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D numpy array with larger integers
    input_dict = {
        "x": np.array([100, 200, 300], dtype=np.int64),
        "loc": np.array([50, 100, 150], dtype=np.int64),
        "scale": np.array([10, 20, 30], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.cdf_3"] = generate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.cdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.cdf_3'.")


check_valid('jax.scipy.stats.cauchy.cdf', generated_inputs['jax.scipy.stats.cauchy.cdf_3'], lib="jax", suffix=3)
