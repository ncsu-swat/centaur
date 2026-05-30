
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ones_inputs():
    list_of_inputs = []

    # Input 1: Float32, small shape
    input_dict = {"shape": 5, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float64, medium shape
    input_dict = {"shape": 10, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Int32, larger shape
    input_dict = {"shape": 100, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int64, minimal shape
    input_dict = {"shape": 1, "dtype": np.int64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Bool, small shape
    input_dict = {"shape": 8, "dtype": np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64
    input_dict = {"shape": 12, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex128
    input_dict = {"shape": 3, "dtype": np.complex128}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int16
    input_dict = {"shape": 15, "dtype": np.int16}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: UInt32
    input_dict = {"shape": 25, "dtype": np.uint32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int8
    input_dict = {"shape": 50, "dtype": np.int8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.ones_1"] = ones_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_1'.")


check_valid('jax.numpy.ones', generated_inputs['jax.numpy.ones_1'], lib="jax", suffix=1)
