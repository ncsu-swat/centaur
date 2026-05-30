
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def logaddexp2_inputs():
    list_of_inputs = []

    # Input 1: Standard python floats
    input_dict = {"x1": 1.5, "x2": 2.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative python floats
    input_dict = {"x1": -3.0, "x2": -4.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zeros
    input_dict = {"x1": 0.0, "x2": 0.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large positive floats
    input_dict = {"x1": 1000.0, "x2": 1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Very small floats (large negative)
    input_dict = {"x1": -1000.0, "x2": -1000.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mixed positive and negative floats
    input_dict = {"x1": 5.0, "x2": -5.0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numpy float32 scalars
    input_dict = {"x1": np.float32(1.23), "x2": np.float32(4.56)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numpy float64 scalars
    input_dict = {"x1": np.float64(-0.123), "x2": np.float64(0.456)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Inf and -Inf
    input_dict = {"x1": np.float64(np.inf), "x2": np.float64(-np.inf)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Very small magnitude floats
    input_dict = {"x1": 1e-15, "x2": 2e-15}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.logaddexp2_2"] = logaddexp2_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.logaddexp2_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.logaddexp2_2'.")


check_valid('jax.numpy.logaddexp2', generated_inputs['jax.numpy.logaddexp2_2'], lib="jax", suffix=2)
