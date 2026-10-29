
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def exp_inputs():
    list_of_inputs = []

    # 1. Zero
    list_of_inputs.append({"x": 0.0})

    # 2. Positive standard float
    list_of_inputs.append({"x": 1.0})

    # 3. Negative standard float
    list_of_inputs.append({"x": -1.0})

    # 4. Small positive float
    list_of_inputs.append({"x": 1e-4})

    # 5. Small negative float
    list_of_inputs.append({"x": -1e-4})

    # 6. Large positive float
    list_of_inputs.append({"x": 50.0})

    # 7. Large negative float
    list_of_inputs.append({"x": -50.0})

    # 8. Fractional float
    list_of_inputs.append({"x": 0.5})

    # 9. Negative fractional float
    list_of_inputs.append({"x": -0.5})

    # 10. Pi
    list_of_inputs.append({"x": float(np.pi)})

    # 11. Infinity
    list_of_inputs.append({"x": float("inf")})

    # 12. Negative infinity
    list_of_inputs.append({"x": float("-inf")})

    # 13. NaN
    list_of_inputs.append({"x": float("nan")})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.exp_2"] = exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp_2'.")


check_valid('jax.numpy.exp', generated_inputs['jax.numpy.exp_2'], lib="jax", suffix=2)
