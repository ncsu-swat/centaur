
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ix_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    input_dict = {"args": np.array([0, 1, 2], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int64 array
    input_dict = {"args": np.array([10, 20, 30, 40, 50], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 array
    input_dict = {"args": np.array([1.5, 2.5, 3.5], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float64 array with negative values
    input_dict = {"args": np.array([-1.0, 0.0, 1.0], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another 1D int32 array
    input_dict = {"args": np.array([5, 10, 15], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int64 array with positive values
    input_dict = {"args": np.array([100, 200, 300], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D int32 array with negative values
    input_dict = {"args": np.array([-10, -20, -30], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another 1D int32 array of larger size
    input_dict = {"args": np.array([0, 1, 2, 3, 4], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D int64 array with mixed values
    input_dict = {"args": np.array([-5, 5], dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array with a single element
    input_dict = {"args": np.array([0.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ix_"] = ix_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ix_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ix_'.")


check_valid('jax.numpy.ix_', generated_inputs['jax.numpy.ix_'], lib="jax", suffix=0)
