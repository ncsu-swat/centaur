
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_full_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 array
    list_of_inputs.append({
        "shape": 5,
        "fill_value": 3.14,
        "dtype": np.dtype('float32')
    })

    # Input 2: Negative fill value
    list_of_inputs.append({
        "shape": 10,
        "fill_value": -1.5,
        "dtype": np.dtype('float32')
    })

    # Input 3: Float64 high precision
    list_of_inputs.append({
        "shape": 100,
        "fill_value": 2.718281828459,
        "dtype": np.dtype('float64')
    })

    # Input 4: Int32 target dtype
    list_of_inputs.append({
        "shape": 8,
        "fill_value": 4.0,
        "dtype": np.dtype('int32')
    })

    # Input 5: Int64 negative target dtype
    list_of_inputs.append({
        "shape": 12,
        "fill_value": -10.0,
        "dtype": np.dtype('int64')
    })

    # Input 6: Float16 low precision
    list_of_inputs.append({
        "shape": 3,
        "fill_value": 0.001,
        "dtype": np.dtype('float16')
    })

    # Input 7: Bool target dtype
    list_of_inputs.append({
        "shape": 15,
        "fill_value": 1.0,
        "dtype": np.dtype('bool')
    })

    # Input 8: NaN fill value
    list_of_inputs.append({
        "shape": 6,
        "fill_value": float('nan'),
        "dtype": np.dtype('float32')
    })

    # Input 9: Inf fill value
    list_of_inputs.append({
        "shape": 4,
        "fill_value": float('inf'),
        "dtype": np.dtype('float64')
    })

    # Input 10: Large size with negative zero
    list_of_inputs.append({
        "shape": 1000,
        "fill_value": -0.0,
        "dtype": np.dtype('float32')
    })

    return list_of_inputs

generated_inputs["jax.numpy.full_2"] = jax_numpy_full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_2'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_2'], lib="jax", suffix=2)
