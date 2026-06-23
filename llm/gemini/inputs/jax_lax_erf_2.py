
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_lax_erf_inputs():
    list_of_inputs = []

    # Input 1: Standard positive float
    list_of_inputs.append({"x": 1.0})

    # Input 2: Standard negative float
    list_of_inputs.append({"x": -1.0})

    # Input 3: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 4: Large positive float
    list_of_inputs.append({"x": 5.0})

    # Input 5: Large negative float
    list_of_inputs.append({"x": -5.0})

    # Input 6: Very small positive float
    list_of_inputs.append({"x": 1e-6})

    # Input 7: Numpy float32 scalar
    list_of_inputs.append({"x": np.float32(0.5)})

    # Input 8: Numpy float64 scalar
    list_of_inputs.append({"x": np.float64(-2.5)})

    # Input 9: Numpy float16 scalar
    list_of_inputs.append({"x": np.float16(1.5)})

    # Input 10: Large numpy float
    list_of_inputs.append({"x": np.float32(10.0)})

    # Input 11: Very small negative numpy float
    list_of_inputs.append({"x": np.float64(-1e-8)})

    return list_of_inputs

generated_inputs["jax.lax.erf_2"] = jax_lax_erf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_2'.")


check_valid('jax.lax.erf', generated_inputs['jax.lax.erf_2'], lib="jax", suffix=2)
