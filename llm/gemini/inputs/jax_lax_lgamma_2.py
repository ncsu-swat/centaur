
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: Python float (positive)
    list_of_inputs.append({"x": 1.5})

    # Input 2: Python float (integer value represented as float)
    list_of_inputs.append({"x": 3.0})

    # Input 3: Python float (small positive value)
    list_of_inputs.append({"x": 0.1})

    # Input 4: Python float (larger positive value)
    list_of_inputs.append({"x": 50.5})

    # Input 5: np.float32 scalar
    list_of_inputs.append({"x": np.float32(2.7)})

    # Input 6: np.float64 scalar
    list_of_inputs.append({"x": np.float64(10.2)})

    # Input 7: np.float32 scalar (small value)
    list_of_inputs.append({"x": np.float32(0.01)})

    # Input 8: np.float64 scalar (large value)
    list_of_inputs.append({"x": np.float64(100.0)})

    # Input 9: Python float (negative non-integer value)
    list_of_inputs.append({"x": -1.5})

    # Input 10: Python float (negative non-integer value)
    list_of_inputs.append({"x": -0.5})

    # Input 11: np.float32 scalar (negative non-integer value)
    list_of_inputs.append({"x": np.float32(-2.5)})

    # Input 12: Python float (very close to zero)
    list_of_inputs.append({"x": 1e-4})

    return list_of_inputs

generated_inputs["jax.lax.lgamma_2"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_2'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_2'], lib="jax", suffix=2)
