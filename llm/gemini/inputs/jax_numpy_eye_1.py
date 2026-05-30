
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eye_inputs():
    list_of_inputs = []

    # Input 1: Basic square float32
    input_dict = {'N': 3, 'M': 3, 'k': 0, 'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-square float32
    input_dict = {'N': 3, 'M': 5, 'k': 0, 'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Square int32 with positive diagonal offset
    input_dict = {'N': 4, 'M': 4, 'k': 1, 'dtype': np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Square int32 with negative diagonal offset
    input_dict = {'N': 4, 'M': 4, 'k': -1, 'dtype': np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular float64 with positive diagonal offset
    input_dict = {'N': 5, 'M': 3, 'k': 2, 'dtype': np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rectangular float64 with negative diagonal offset
    input_dict = {'N': 3, 'M': 6, 'k': -2, 'dtype': np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean dtype
    input_dict = {'N': 3, 'M': 3, 'k': 0, 'dtype': np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex64 square matrix
    input_dict = {'N': 5, 'M': 5, 'k': 0, 'dtype': np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large dimensions, int64 with offset
    input_dict = {'N': 100, 'M': 100, 'k': 10, 'dtype': np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Small 1x1 matrix
    input_dict = {'N': 1, 'M': 1, 'k': 0, 'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Extreme diagonal offset
    input_dict = {'N': 3, 'M': 3, 'k': 5, 'dtype': np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.eye_1"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.eye_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.eye_1'.")


check_valid('jax.numpy.eye', generated_inputs['jax.numpy.eye_1'], lib="jax", suffix=1)
