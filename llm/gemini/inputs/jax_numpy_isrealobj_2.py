
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Basic integer 0
    list_of_inputs.append({"x": 0})

    # Input 2: Positive integer
    list_of_inputs.append({"x": 42})

    # Input 3: Negative integer
    list_of_inputs.append({"x": -100})

    # Input 4: Large integer
    list_of_inputs.append({"x": 1234567890})

    # Input 5: numpy int32 scalar
    list_of_inputs.append({"x": np.int32(15)})

    # Input 6: numpy int64 scalar
    list_of_inputs.append({"x": np.int64(-9999)})

    # Input 7: numpy uint8 scalar
    list_of_inputs.append({"x": np.uint8(255)})

    # Input 8: numpy int16 scalar
    list_of_inputs.append({"x": np.int16(32000)})

    # Input 9: Negative numpy int8 scalar
    list_of_inputs.append({"x": np.int8(-128)})

    # Input 10: Another positive integer
    list_of_inputs.append({"x": 100000})

    return list_of_inputs

generated_inputs["jax.numpy.isrealobj_2"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_2'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_2'], lib="jax", suffix=2)
