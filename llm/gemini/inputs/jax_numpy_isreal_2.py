
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []

    # Input 1: positive standard python integer
    input_dict = {"x": 5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative standard python integer
    input_dict = {"x": -10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero
    input_dict = {"x": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: large python integer
    input_dict = {"x": 123456789}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy int32
    input_dict = {"x": np.int32(42)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: numpy int64 negative
    input_dict = {"x": np.int64(-999)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: numpy int16 positive
    input_dict = {"x": np.int16(15)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: numpy uint8 positive
    input_dict = {"x": np.uint8(255)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: numpy int8 negative
    input_dict = {"x": np.int8(-128)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: numpy intc
    input_dict = {"x": np.intc(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isreal_2"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_2'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_2'], lib="jax", suffix=2)
