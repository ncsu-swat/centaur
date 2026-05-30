
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asarray_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        "a": 0,
        "dtype": np.int32,
        "order": "K",
        "copy": True
    })

    # Input 2
    list_of_inputs.append({
        "a": 42,
        "dtype": np.float32,
        "order": "K",
        "copy": False
    })

    # Input 3
    list_of_inputs.append({
        "a": -100,
        "dtype": np.int64,
        "order": "K",
        "copy": True
    })

    # Input 4
    list_of_inputs.append({
        "a": 2024,
        "dtype": np.float64,
        "order": "K",
        "copy": False
    })

    # Input 5
    list_of_inputs.append({
        "a": -1,
        "dtype": np.int16,
        "order": "K",
        "copy": True
    })

    # Input 6
    list_of_inputs.append({
        "a": 100000,
        "dtype": np.uint32,
        "order": "K",
        "copy": False
    })

    # Input 7
    list_of_inputs.append({
        "a": 5,
        "dtype": np.bool_,
        "order": "K",
        "copy": True
    })

    # Input 8
    list_of_inputs.append({
        "a": -999,
        "dtype": np.complex64,
        "order": "K",
        "copy": False
    })

    # Input 9
    list_of_inputs.append({
        "a": 123456,
        "dtype": np.float16,
        "order": "K",
        "copy": True
    })

    # Input 10
    list_of_inputs.append({
        "a": 7,
        "dtype": np.uint8,
        "order": "K",
        "copy": False
    })

    return list_of_inputs

generated_inputs["jax.numpy.asarray_4"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_4'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_4'], lib="jax", suffix=4)
