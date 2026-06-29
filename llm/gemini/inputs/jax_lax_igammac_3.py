
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def igammac_inputs():
    list_of_inputs = []

    # Input 1: float with integer value
    list_of_inputs.append({"a": 1.0, "x": 1.0})

    # Input 2: float with integer value
    list_of_inputs.append({"a": 2.0, "x": 5.0})

    # Input 3: np.float32 with integer values
    list_of_inputs.append({"a": np.float32(3.0), "x": np.float32(2.0)})

    # Input 4: np.float64 with integer values
    list_of_inputs.append({"a": np.float64(5.0), "x": np.float64(10.0)})

    # Input 5: float with integer value
    list_of_inputs.append({"a": 4.0, "x": 1.0})

    # Input 6: x is 0.0
    list_of_inputs.append({"a": 10.0, "x": 0.0})

    # Input 7: np.float32 with integer values
    list_of_inputs.append({"a": np.float32(2.0), "x": np.float32(3.0)})

    # Input 8: Larger integer values
    list_of_inputs.append({"a": 8.0, "x": 12.0})

    # Input 9: np.float64 with x as 0.0
    list_of_inputs.append({"a": np.float64(1.0), "x": np.float64(0.0)})

    # Input 10: Equal integer values
    list_of_inputs.append({"a": 15.0, "x": 15.0})

    return list_of_inputs

generated_inputs["jax.lax.igammac_3"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igammac_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igammac_3'.")


check_valid('jax.lax.igammac', generated_inputs['jax.lax.igammac_3'], lib="jax", suffix=3)
