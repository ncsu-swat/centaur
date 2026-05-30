
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: positive python integer
    list_of_inputs.append({"x": 5})

    # Input 2: negative python integer
    list_of_inputs.append({"x": -5})

    # Input 3: zero python integer
    list_of_inputs.append({"x": 0})

    # Input 4: np.int32 positive
    list_of_inputs.append({"x": np.int32(10)})

    # Input 5: np.int32 negative
    list_of_inputs.append({"x": np.int32(-10)})

    # Input 6: np.int64 large positive
    list_of_inputs.append({"x": np.int64(100000)})

    # Input 7: np.int64 large negative
    list_of_inputs.append({"x": np.int64(-100000)})

    # Input 8: np.int16 positive
    list_of_inputs.append({"x": np.int16(123)})

    # Input 9: np.int8 positive boundary
    list_of_inputs.append({"x": np.int8(127)})

    # Input 10: np.int8 negative boundary
    list_of_inputs.append({"x": np.int8(-128)})

    return list_of_inputs

generated_inputs["jax.numpy.asinh_3"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_3'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_3'], lib="jax", suffix=3)
