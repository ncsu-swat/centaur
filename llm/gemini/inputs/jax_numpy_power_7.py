
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: Basic positive floats
    list_of_inputs.append({"x1": 2.0, "x2": 3.0})

    # Input 2: Fractional exponent (square root)
    list_of_inputs.append({"x1": 16.0, "x2": 0.5})

    # Input 3: Negative exponent
    list_of_inputs.append({"x1": 4.0, "x2": -2.0})

    # Input 4: Exponent zero
    list_of_inputs.append({"x1": 5.5, "x2": 0.0})

    # Input 5: Zero base with positive exponent
    list_of_inputs.append({"x1": 0.0, "x2": 2.5})

    # Input 6: Negative base with odd integer-valued float exponent
    list_of_inputs.append({"x1": -2.0, "x2": 3.0})

    # Input 7: Negative base with even integer-valued float exponent
    list_of_inputs.append({"x1": -3.0, "x2": 2.0})

    # Input 8: Base less than 1 with positive exponent
    list_of_inputs.append({"x1": 0.5, "x2": 4.0})

    # Input 9: Fractional base with negative exponent
    list_of_inputs.append({"x1": 0.25, "x2": -1.5})

    # Input 10: Both base and exponent fractional
    list_of_inputs.append({"x1": 2.5, "x2": 1.5})

    # Input 11: Base 1.0 with large exponent
    list_of_inputs.append({"x1": 1.0, "x2": 100.0})

    # Input 12: Negative base with negative integer-valued float exponent
    list_of_inputs.append({"x1": -2.0, "x2": -2.0})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.power_7"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_7'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_7'], lib="jax", suffix=7)
