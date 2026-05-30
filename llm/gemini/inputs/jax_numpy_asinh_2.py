
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def asinh_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Standard negative float
    list_of_inputs.append({"x": -1.0})

    # Input 3: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: Large float
    list_of_inputs.append({"x": 1e6})

    # Input 5: Very small float
    list_of_inputs.append({"x": 1e-6})

    # Input 6: numpy float32 positive scalar
    list_of_inputs.append({"x": np.float32(2.5)})

    # Input 7: numpy float32 negative scalar
    list_of_inputs.append({"x": np.float32(-3.5)})

    # Input 8: numpy float64 positive scalar
    list_of_inputs.append({"x": np.float64(10.2)})

    # Input 9: numpy float16 positive scalar
    list_of_inputs.append({"x": np.float16(0.5)})

    # Input 10: Positive infinity
    list_of_inputs.append({"x": float('inf')})

    # Input 11: Negative infinity
    list_of_inputs.append({"x": float('-inf')})

    return list_of_inputs

generated_inputs["jax.numpy.asinh_2"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_2'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_2'], lib="jax", suffix=2)
