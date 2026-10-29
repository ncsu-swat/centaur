
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def isrealobj_inputs():
    list_of_inputs = []

    # Input 1: Python True
    list_of_inputs.append({"x": True})

    # Input 2: Python False
    list_of_inputs.append({"x": False})

    # Input 3: NumPy bool_ True
    list_of_inputs.append({"x": np.bool_(True)})

    # Input 4: NumPy bool_ False
    list_of_inputs.append({"x": np.bool_(False)})

    # Input 5: NumPy np.True_ singleton
    list_of_inputs.append({"x": np.True_})

    # Input 6: NumPy np.False_ singleton
    list_of_inputs.append({"x": np.False_})

    # Input 7: Boolean from comparison True
    list_of_inputs.append({"x": bool(1 < 2)})

    # Input 8: Boolean from comparison False
    list_of_inputs.append({"x": bool(1 > 2)})

    # Input 9: NumPy bool cast from int
    list_of_inputs.append({"x": np.bool_(1)})

    # Input 10: NumPy bool cast from zero
    list_of_inputs.append({"x": np.bool_(0)})

    # Input 11: Python bool from int
    list_of_inputs.append({"x": bool(0)})

    # Input 12: Python bool from negative int
    list_of_inputs.append({"x": bool(-5)})

    return [copy.deepcopy(x) for x in list_of_inputs]


generated_inputs["jax.numpy.isrealobj_4"] = isrealobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isrealobj_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isrealobj_4'.")


check_valid('jax.numpy.isrealobj', generated_inputs['jax.numpy.isrealobj_4'], lib="jax", suffix=4)
