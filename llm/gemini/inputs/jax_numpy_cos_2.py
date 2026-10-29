
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def cos_inputs():
    list_of_inputs = []

    # Input 1: Zero
    list_of_inputs.append({"x": 0.0})

    # Input 2: Positive pi / 2
    list_of_inputs.append({"x": float(np.pi / 2)})

    # Input 3: Positive pi
    list_of_inputs.append({"x": float(np.pi)})

    # Input 4: Negative angle
    list_of_inputs.append({"x": float(-np.pi / 4)})

    # Input 5: Small positive float
    list_of_inputs.append({"x": 1e-6})

    # Input 6: Large positive float
    list_of_inputs.append({"x": 1e4})

    # Input 7: Arbitrary negative float
    list_of_inputs.append({"x": -2.718281828})

    # Input 8: Two pi
    list_of_inputs.append({"x": float(2 * np.pi)})

    # Input 9: Negative zero
    list_of_inputs.append({"x": -0.0})

    # Input 10: 3 * pi / 2
    list_of_inputs.append({"x": float(3 * np.pi / 2)})

    # Input 11: High magnitude negative float
    list_of_inputs.append({"x": -12345.6789})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.cos_2"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cos_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cos_2'.")


check_valid('jax.numpy.cos', generated_inputs['jax.numpy.cos_2'], lib="jax", suffix=2)
