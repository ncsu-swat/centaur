
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isscalar_inputs():
    list_of_inputs = []

    # 1. int32
    input_dict = {"element": np.dtype('int32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2. float32
    input_dict = {"element": np.dtype('float32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3. bool
    input_dict = {"element": np.dtype('bool')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4. complex64
    input_dict = {"element": np.dtype('complex64')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5. int8
    input_dict = {"element": np.dtype('int8')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6. int16
    input_dict = {"element": np.dtype('int16')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7. uint8
    input_dict = {"element": np.dtype('uint8')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8. uint16
    input_dict = {"element": np.dtype('uint16')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9. uint32
    input_dict = {"element": np.dtype('uint32')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10. float16
    input_dict = {"element": np.dtype('float16')}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isscalar_8"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_8'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_8'], lib="jax", suffix=8)
