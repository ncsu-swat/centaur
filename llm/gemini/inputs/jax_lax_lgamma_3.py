
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lgamma_inputs():
    list_of_inputs = []

    # Input 1: float32 representing integer 5
    list_of_inputs.append({"x": np.float32(5.0)})

    # Input 2: float64 representing integer 10
    list_of_inputs.append({"x": np.float64(10.0)})

    # Input 3: float32 representing integer 3
    list_of_inputs.append({"x": np.float32(3.0)})

    # Input 4: float64 representing integer 8
    list_of_inputs.append({"x": np.float64(8.0)})

    # Input 5: float32 representing integer 12
    list_of_inputs.append({"x": np.float32(12.0)})

    # Input 6: float64 representing integer 4
    list_of_inputs.append({"x": np.float64(4.0)})

    # Input 7: float32 representing integer 6
    list_of_inputs.append({"x": np.float32(6.0)})

    # Input 8: float64 representing integer 15
    list_of_inputs.append({"x": np.float64(15.0)})

    # Input 9: float32 representing integer 100
    list_of_inputs.append({"x": np.float32(100.0)})

    # Input 10: float64 representing integer 1
    list_of_inputs.append({"x": np.float64(1.0)})

    # Input 11: float32 representing integer 2
    list_of_inputs.append({"x": np.float32(2.0)})

    return list_of_inputs

generated_inputs["jax.lax.lgamma_3"] = lgamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.lgamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.lgamma_3'.")


check_valid('jax.lax.lgamma', generated_inputs['jax.lax.lgamma_3'], lib="jax", suffix=3)
