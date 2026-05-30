
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isinf_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Standard positive float
    list_of_inputs.append({"x": 12.34})

    # Input 3: Standard negative float
    list_of_inputs.append({"x": -56.78})

    # Input 4: Positive infinity as Python float
    list_of_inputs.append({"x": float('inf')})

    # Input 5: Negative infinity as Python float
    list_of_inputs.append({"x": float('-inf')})

    # Input 6: NaN as Python float
    list_of_inputs.append({"x": float('nan')})

    # Input 7: Large scientific notation float
    list_of_inputs.append({"x": 1e308})

    # Input 8: Small scientific notation float
    list_of_inputs.append({"x": -1e-308})

    # Input 9: Positive infinity as numpy float64
    list_of_inputs.append({"x": np.float64(np.inf)})

    # Input 10: Negative infinity as numpy float32
    list_of_inputs.append({"x": np.float32(-np.inf)})

    # Input 11: Standard float as numpy float64
    list_of_inputs.append({"x": np.float64(3.14159)})

    return list_of_inputs

generated_inputs["jax.numpy.isinf_2"] = isinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isinf_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isinf_2'.")


check_valid('jax.numpy.isinf', generated_inputs['jax.numpy.isinf_2'], lib="jax", suffix=2)
