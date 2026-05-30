
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Python int
    list_of_inputs.append({"element": 42})

    # Input 2: Python int (negative)
    list_of_inputs.append({"element": -17})

    # Input 3: Python int (zero)
    list_of_inputs.append({"element": 0})

    # Input 4: numpy int32
    list_of_inputs.append({"element": np.int32(100)})

    # Input 5: numpy int32 (negative)
    list_of_inputs.append({"element": np.int32(-50)})

    # Input 6: numpy int64
    list_of_inputs.append({"element": np.int64(12345678)})

    # Input 7: numpy int64 (negative)
    list_of_inputs.append({"element": np.int64(-12345678)})

    # Input 8: Python int (large positive)
    list_of_inputs.append({"element": 99999999})

    # Input 9: Python int (large negative)
    list_of_inputs.append({"element": -99999999})

    # Input 10: numpy int32 (zero)
    list_of_inputs.append({"element": np.int32(0)})

    # Input 11: numpy int64 (zero)
    list_of_inputs.append({"element": np.int64(0)})

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_2"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_2'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_2'], lib="jax", suffix=2)
