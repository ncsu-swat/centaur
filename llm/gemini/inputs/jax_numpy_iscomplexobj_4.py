
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def iscomplexobj_inputs():
    list_of_inputs = []

    # Input 1: Python standard boolean True
    list_of_inputs.append(copy.deepcopy({"x": True}))

    # Input 2: Python standard boolean False
    list_of_inputs.append(copy.deepcopy({"x": False}))

    # Input 3: NumPy bool_ True
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(True)}))

    # Input 4: NumPy bool_ False
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(False)}))

    # Input 5: Python bool from int 1
    list_of_inputs.append(copy.deepcopy({"x": bool(1)}))

    # Input 6: Python bool from int 0
    list_of_inputs.append(copy.deepcopy({"x": bool(0)}))

    # Input 7: NumPy bool_ from integer 1
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(1)}))

    # Input 8: NumPy bool_ from integer 0
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(0)}))

    # Input 9: Python bool from expression
    list_of_inputs.append(copy.deepcopy({"x": bool(1 > 2)}))

    # Input 10: Python bool from expression
    list_of_inputs.append(copy.deepcopy({"x": bool(2 > 1)}))

    # Input 11: NumPy bool_ from float comparison
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(0.0 != 0.0)}))

    # Input 12: NumPy bool_ from float comparison
    list_of_inputs.append(copy.deepcopy({"x": np.bool_(1.0 == 1.0)}))

    return list_of_inputs


generated_inputs["jax.numpy.iscomplexobj_4"] = iscomplexobj_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.iscomplexobj_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.iscomplexobj_4'.")


check_valid('jax.numpy.iscomplexobj', generated_inputs['jax.numpy.iscomplexobj_4'], lib="jax", suffix=4)
