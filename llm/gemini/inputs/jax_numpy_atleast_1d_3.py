
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    input_dict = {"arys": 1.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Standard negative float
    input_dict = {"arys": -5.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero float
    input_dict = {"arys": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large float
    input_dict = {"arys": 1.23e5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small float
    input_dict = {"arys": 4.56e-6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy float64 positive
    input_dict = {"arys": np.float64(10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy float32 negative
    input_dict = {"arys": np.float32(-10.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy float16
    input_dict = {"arys": np.float16(0.5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Positive infinity
    input_dict = {"arys": float('inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative infinity
    input_dict = {"arys": float('-inf')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: NaN (Not a Number)
    input_dict = {"arys": float('nan')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: numpy float64 small negative
    input_dict = {"arys": np.float64(-3.14e-2)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.atleast_1d_3"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_3'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_3'], lib="jax", suffix=3)
