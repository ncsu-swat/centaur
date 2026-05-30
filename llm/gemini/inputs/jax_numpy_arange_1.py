
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arange_inputs():
    list_of_inputs = []

    # Input 1: Simple range
    list_of_inputs.append({
        "start": 0,
        "stop": 10,
        "step": 1,
        "dtype": np.dtype('int32')
    })

    # Input 2: Step of 2
    list_of_inputs.append({
        "start": 1,
        "stop": 10,
        "step": 2,
        "dtype": np.dtype('int64')
    })

    # Input 3: Negative start
    list_of_inputs.append({
        "start": -10,
        "stop": 0,
        "step": 1,
        "dtype": np.dtype('int32')
    })

    # Input 4: Negative step
    list_of_inputs.append({
        "start": 10,
        "stop": 0,
        "step": -1,
        "dtype": np.dtype('int32')
    })

    # Input 5: Float32 output dtype
    list_of_inputs.append({
        "start": -5,
        "stop": 5,
        "step": 2,
        "dtype": np.dtype('float32')
    })

    # Input 6: Large step
    list_of_inputs.append({
        "start": 0,
        "stop": 100,
        "step": 10,
        "dtype": np.dtype('int32')
    })

    # Input 7: Negative start, stop, and step
    list_of_inputs.append({
        "start": -2,
        "stop": -12,
        "step": -2,
        "dtype": np.dtype('int16')
    })

    # Input 8: Float64 output dtype
    list_of_inputs.append({
        "start": -100,
        "stop": -50,
        "step": 5,
        "dtype": np.dtype('float64')
    })

    # Input 9: Unsigned int dtype
    list_of_inputs.append({
        "start": 5,
        "stop": 15,
        "step": 3,
        "dtype": np.dtype('uint32')
    })

    # Input 10: Downwards with int64
    list_of_inputs.append({
        "start": -10,
        "stop": -20,
        "step": -3,
        "dtype": np.dtype('int64')
    })

    return list_of_inputs

generated_inputs["jax.numpy.arange_1"] = arange_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arange_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arange_1'.")


check_valid('jax.numpy.arange', generated_inputs['jax.numpy.arange_1'], lib="jax", suffix=1)
