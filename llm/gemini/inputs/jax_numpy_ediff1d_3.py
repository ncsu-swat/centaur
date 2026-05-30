
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    ary = np.array([1.0, 2.5, 4.0, 7.0], dtype=np.float32)
    input_dict = {
        "ary": ary,
        "to_end": 10.0,
        "to_begin": -5.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array with negative and positive values
    ary = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "ary": ary,
        "to_end": 0.0,
        "to_begin": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float32 array with random values
    ary = np.random.randn(2, 3, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "to_end": -1.5,
        "to_begin": 2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with larger values
    ary = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    input_dict = {
        "ary": ary,
        "to_end": 100.0,
        "to_begin": -100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 array
    ary = np.random.randn(3, 4).astype(np.float64)
    input_dict = {
        "ary": ary,
        "to_end": 0.5,
        "to_begin": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 array (as tensor input)
    ary = np.array([5, 10, 15, 20], dtype=np.int32)
    input_dict = {
        "ary": ary,
        "to_end": 25.0,
        "to_begin": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    ary = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "ary": ary,
        "to_end": 9.9,
        "to_begin": -9.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small 1D float32 array
    ary = np.array([1.5, 3.0], dtype=np.float32)
    input_dict = {
        "ary": ary,
        "to_end": 4.5,
        "to_begin": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element 1D float32 array
    ary = np.array([42.0], dtype=np.float32)
    input_dict = {
        "ary": ary,
        "to_end": 1.0,
        "to_begin": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int64 array
    ary = np.array([[100, 200], [300, 400]], dtype=np.int64)
    input_dict = {
        "ary": ary,
        "to_end": 500.0,
        "to_begin": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ediff1d_3"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_3'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_3'], lib="jax", suffix=3)
