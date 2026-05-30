
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isnan_inputs():
    list_of_inputs = []

    # Input 1: numpy float64 positive
    list_of_inputs.append({"x": np.float64(3.14)})

    # Input 2: numpy float32 negative
    list_of_inputs.append({"x": np.float32(-2.718)})

    # Input 3: numpy float64 zero
    list_of_inputs.append({"x": np.float64(0.0)})

    # Input 4: numpy float32 negative zero
    list_of_inputs.append({"x": np.float32(-0.0)})

    # Input 5: numpy float64 NaN
    list_of_inputs.append({"x": np.float64(np.nan)})

    # Input 6: numpy float32 NaN
    list_of_inputs.append({"x": np.float32(np.nan)})

    # Input 7: numpy float64 infinity
    list_of_inputs.append({"x": np.float64(np.inf)})

    # Input 8: numpy float32 negative infinity
    list_of_inputs.append({"x": np.float32(-np.inf)})

    # Input 9: Python native float positive
    list_of_inputs.append({"x": 1.23e5})

    # Input 10: Python native float negative
    list_of_inputs.append({"x": -4.56e-6})

    return list_of_inputs

generated_inputs["jax.numpy.isnan_2"] = isnan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isnan_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isnan_2'.")


check_valid('jax.numpy.isnan', generated_inputs['jax.numpy.isnan_2'], lib="jax", suffix=2)
