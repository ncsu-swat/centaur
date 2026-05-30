
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_issubdtype_inputs():
    list_of_inputs = []

    # Input 1
    list_of_inputs.append({
        'arg1': np.dtype('int32'),
        'arg2': 'int32'
    })

    # Input 2
    list_of_inputs.append({
        'arg1': np.dtype('int32'),
        'arg2': 'int64'
    })

    # Input 3
    list_of_inputs.append({
        'arg1': np.dtype('float64'),
        'arg2': 'float64'
    })

    # Input 4
    list_of_inputs.append({
        'arg1': np.dtype('float64'),
        'arg2': 'float32'
    })

    # Input 5
    list_of_inputs.append({
        'arg1': np.dtype('uint8'),
        'arg2': 'uint8'
    })

    # Input 6
    list_of_inputs.append({
        'arg1': np.dtype('uint8'),
        'arg2': 'int8'
    })

    # Input 7
    list_of_inputs.append({
        'arg1': np.dtype('bool'),
        'arg2': 'bool'
    })

    # Input 8
    list_of_inputs.append({
        'arg1': np.dtype('complex64'),
        'arg2': 'complex64'
    })

    # Input 9
    list_of_inputs.append({
        'arg1': np.dtype('int16'),
        'arg2': 'int16'
    })

    # Input 10
    list_of_inputs.append({
        'arg1': np.dtype('float32'),
        'arg2': 'float32'
    })

    # Input 11
    list_of_inputs.append({
        'arg1': np.dtype('uint32'),
        'arg2': 'uint32'
    })

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.issubdtype_3"] = jax_numpy_issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_3'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_3'], lib="jax", suffix=3)
