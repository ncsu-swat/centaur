
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_gumbel_r_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Scalar Python ints
    input_dict = {
        'x': int(2),
        'loc': int(0),
        'scale': int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: NumPy scalar ints
    input_dict = {
        'x': np.int32(-1),
        'loc': np.int32(-2),
        'scale': np.int32(3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D arrays of np.int32
    input_dict = {
        'x': np.array([-5, 0, 5], dtype=np.int32),
        'loc': np.array([-1, 0, 1], dtype=np.int32),
        'scale': np.array([2, 2, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays of np.int64
    input_dict = {
        'x': np.array([[1, 2], [3, 4]], dtype=np.int64),
        'loc': np.array([[0, 1], [0, 1]], dtype=np.int64),
        'scale': np.array([[2, 3], [4, 5]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broad-casting with scalar loc and scale
    input_dict = {
        'x': np.array([1, 2, 3, 4], dtype=np.int32),
        'loc': np.int32(0),
        'scale': np.int32(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broad-casting with 1D x and 2D loc, scale
    input_dict = {
        'x': np.array([1, 2], dtype=np.int32),
        'loc': np.array([[0, 0], [1, 1]], dtype=np.int32),
        'scale': np.array([[1, 2], [1, 2]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large integers (np.int64)
    input_dict = {
        'x': np.array([100, -100], dtype=np.int64),
        'loc': np.array([10, -10], dtype=np.int64),
        'scale': np.array([5, 5], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays
    input_dict = {
        'x': np.ones((2, 2, 2), dtype=np.int32) * 5,
        'loc': np.ones((2, 2, 2), dtype=np.int32) * 2,
        'scale': np.ones((2, 2, 2), dtype=np.int32) * 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Python int for scale and arrays for x and loc
    input_dict = {
        'x': np.array([3, 4, 5], dtype=np.int32),
        'loc': np.array([1, 1, 1], dtype=np.int32),
        'scale': int(2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixing np.int64 and np.int32 types
    input_dict = {
        'x': np.array([10], dtype=np.int64),
        'loc': np.array([2], dtype=np.int32),
        'scale': np.array([3], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logcdf_3"] = generate_gumbel_r_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logcdf_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logcdf_3'.")


check_valid('jax.scipy.stats.gumbel_r.logcdf', generated_inputs['jax.scipy.stats.gumbel_r.logcdf_3'], lib="jax", suffix=3)
