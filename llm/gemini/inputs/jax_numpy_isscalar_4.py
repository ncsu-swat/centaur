
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isscalar_inputs():
    list_of_inputs = []

    # Input 1: Python True
    list_of_inputs.append({"element": True})

    # Input 2: Python False
    list_of_inputs.append({"element": False})

    # Input 3: numpy bool True
    list_of_inputs.append({"element": np.bool_(True)})

    # Input 4: numpy bool False
    list_of_inputs.append({"element": np.bool_(False)})

    # Input 5: numpy True_
    list_of_inputs.append({"element": np.True_})

    # Input 6: numpy False_
    list_of_inputs.append({"element": np.False_})

    # Input 7: numpy bool from integer 1
    list_of_inputs.append({"element": np.bool_(1)})

    # Input 8: numpy bool from integer 0
    list_of_inputs.append({"element": np.bool_(0)})

    # Input 9: Python bool constructed from comparison (True)
    list_of_inputs.append({"element": bool(1 < 2)})

    # Input 10: Python bool constructed from comparison (False)
    list_of_inputs.append({"element": bool(1 > 2)})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.isscalar_4"] = isscalar_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isscalar_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isscalar_4'.")


check_valid('jax.numpy.isscalar', generated_inputs['jax.numpy.isscalar_4'], lib="jax", suffix=4)
