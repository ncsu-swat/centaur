
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def erfc_inputs():
    list_of_inputs = []

    # Since jax.lax.erfc strictly requires floating subtypes, boolean values
    # must be represented as floating-point representations of booleans (True -> 1.0, False -> 0.0)
    # Input 1: float32 True
    list_of_inputs.append({"x": np.float32(True)})

    # Input 2: float32 False
    list_of_inputs.append({"x": np.float32(False)})

    # Input 3: float64 True
    list_of_inputs.append({"x": np.float64(True)})

    # Input 4: float64 False
    list_of_inputs.append({"x": np.float64(False)})

    # Input 5: Python float True
    list_of_inputs.append({"x": float(True)})

    # Input 6: Python float False
    list_of_inputs.append({"x": float(False)})

    # Input 7: float32 from expression (True)
    list_of_inputs.append({"x": np.float32(10 > 5)})

    # Input 8: float32 from expression (False)
    list_of_inputs.append({"x": np.float32(5 > 10)})

    # Input 9: float64 from expression (True)
    list_of_inputs.append({"x": np.float64(1.0 != 0.0)})

    # Input 10: float64 from expression (False)
    list_of_inputs.append({"x": np.float64(1.0 == 0.0)})

    # Input 11: Python float literal 1.0
    list_of_inputs.append({"x": 1.0})

    # Input 12: Python float literal 0.0
    list_of_inputs.append({"x": 0.0})

    return list_of_inputs


generated_inputs["jax.lax.erfc_4"] = erfc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erfc_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erfc_4'.")


check_valid('jax.lax.erfc', generated_inputs['jax.lax.erfc_4'], lib="jax", suffix=4)
