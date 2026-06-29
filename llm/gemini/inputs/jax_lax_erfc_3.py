
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erfc_inputs():
    list_of_inputs = []

    # Input 1: positive float
    list_of_inputs.append({"x": 0.5})

    # Input 2: negative float
    list_of_inputs.append({"x": -0.5})

    # Input 3: zero float
    list_of_inputs.append({"x": 0.0})

    # Input 4: large positive float
    list_of_inputs.append({"x": 5.0})

    # Input 5: large negative float
    list_of_inputs.append({"x": -5.0})

    # Input 6: positive np.float32 scalar
    list_of_inputs.append({"x": np.float32(1.23)})

    # Input 7: negative np.float32 scalar
    list_of_inputs.append({"x": np.float32(-1.23)})

    # Input 8: positive np.float64 scalar
    list_of_inputs.append({"x": np.float64(3.14159)})

    # Input 9: negative np.float64 scalar
    list_of_inputs.append({"x": np.float64(-3.14159)})

    # Input 10: small positive float
    list_of_inputs.append({"x": 1e-6})

    # Input 11: small negative float
    list_of_inputs.append({"x": -1e-6})

    # Input 12: np.float32 zero scalar
    list_of_inputs.append({"x": np.float32(0.0)})

    return list_of_inputs

generated_inputs["jax.lax.erfc_3"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_3'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_3'], lib="jax", suffix=3)
