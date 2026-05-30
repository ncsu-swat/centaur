
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def radians_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0})

    # Input 2: Positive acute angle
    list_of_inputs.append({"x": 45})

    # Input 3: Right angle
    list_of_inputs.append({"x": 90})

    # Input 4: Straight angle
    list_of_inputs.append({"x": 180})

    # Input 5: Full rotation
    list_of_inputs.append({"x": 360})

    # Input 6: Negative angle
    list_of_inputs.append({"x": -90})

    # Input 7: Large negative angle
    list_of_inputs.append({"x": -720})

    # Input 8: Numpy int32 scalar
    list_of_inputs.append({"x": np.int32(270)})

    # Input 9: Numpy int64 scalar
    list_of_inputs.append({"x": np.int64(-135)})

    # Input 10: Numpy int16 scalar
    list_of_inputs.append({"x": np.int16(30)})

    # Input 11: Large positive angle
    list_of_inputs.append({"x": 1080})

    return list_of_inputs

generated_inputs["jax.numpy.radians_5"] = radians_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.radians_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.radians_5'.")


check_valid('jax.numpy.radians', generated_inputs['jax.numpy.radians_5'], lib="jax", suffix=5)
