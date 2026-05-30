
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def result_type_inputs():
    list_of_inputs = []

    # Input 1: int32 dtype
    input_dict = {"args": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32 dtype
    input_dict = {"args": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int64 dtype
    input_dict = {"args": np.dtype('int64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64 dtype
    input_dict = {"args": np.dtype('float64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool dtype
    input_dict = {"args": np.dtype('bool')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: uint8 dtype
    input_dict = {"args": np.dtype('uint8')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint32 dtype
    input_dict = {"args": np.dtype('uint32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64 dtype
    input_dict = {"args": np.dtype('complex64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 dtype
    input_dict = {"args": np.dtype('complex128')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int16 dtype
    input_dict = {"args": np.dtype('int16')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float16 dtype
    input_dict = {"args": np.dtype('float16')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.result_type_1"] = result_type_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.result_type_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.result_type_1'.")


check_valid('jax.numpy.result_type', generated_inputs['jax.numpy.result_type_1'], lib="jax", suffix=1)
