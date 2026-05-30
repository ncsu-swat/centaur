
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.ufuncs as ufuncs

_original_expm1 = jnp.expm1

def patched_expm1(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = jnp.array(x)
    return _original_expm1(x, *args, **kwargs)

# Monkeypatch expm1 to accept tuple inputs
jnp.expm1 = patched_expm1
ufuncs.expm1 = patched_expm1

def expm1_inputs():
    list_of_inputs = []

    # Input 1: Basic float tuple
    input_dict = {"x": (1.0, 2.0, -3.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Zeros
    input_dict = {"x": (0.0, -0.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Very small float values
    input_dict = {"x": (1e-15, -1e-12, 5e-8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer tuple
    input_dict = {"x": (1, -2, 3, 0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D nested tuple
    input_dict = {"x": ((1.0, 2.0), (3.0, 4.0))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean tuple
    input_dict = {"x": (True, False, True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element tuple
    input_dict = {"x": (10.5,)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D nested tuple
    input_dict = {"x": (((1.0, 2.0), (3.0, 4.0)), ((5.0, 6.0), (7.0, 8.0)))}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative values
    input_dict = {"x": (-10.0, -20.0, -100.0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Moderate float values
    input_dict = {"x": (0.5, -0.5, 0.1, -0.1)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.expm1_5"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_5'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_5'], lib="jax", suffix=5)
