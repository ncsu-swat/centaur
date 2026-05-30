
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: Basic python integer 0
    list_of_inputs.append({"x": 0})

    # Input 2: Python integer 1
    list_of_inputs.append({"x": 1})

    # Input 3: Larger positive python integer
    list_of_inputs.append({"x": 100})

    # Input 4: numpy int32 0
    list_of_inputs.append({"x": np.int32(0)})

    # Input 5: numpy int32 5
    list_of_inputs.append({"x": np.int32(5)})

    # Input 6: numpy int64 100
    list_of_inputs.append({"x": np.int64(100)})

    # Input 7: numpy int64 1000
    list_of_inputs.append({"x": np.int64(1000)})

    # Input 8: Python integer 10
    list_of_inputs.append({"x": 10})

    # Input 9: Large python integer
    list_of_inputs.append({"x": 1000000})

    # Input 10: Edge case python integer -1
    list_of_inputs.append({"x": -1})

    # Input 11: numpy int32 -1
    list_of_inputs.append({"x": np.int32(-1)})

    # Input 12: Python integer 42
    list_of_inputs.append({"x": 42})

    return list_of_inputs

generated_inputs["jax.numpy.log1p_3"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log1p_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log1p_3'.")


check_valid('jax.numpy.log1p', generated_inputs['jax.numpy.log1p_3'], lib="jax", suffix=3)
