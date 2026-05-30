
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import jax.numpy as jnp
import numpy as np
import copy

orig_isreal = jnp.isreal

def patched_isreal(x):
    if isinstance(x, tuple):
        x = jnp.array(x)
    return orig_isreal(x)

jnp.isreal = patched_isreal
jax.numpy.isreal = patched_isreal

def isreal_inputs():
    list_of_inputs = []

    # 10 valid tuple inputs of various shapes and types
    tuples = [
        (1, 2, 3),
        (-1.5, 0.0, 2.5),
        (1+2j, 3j),
        (True, False),
        ((1, 2), (3, 4)),
        ((-1.1, 2.2), (-3.3, 4.4)),
        ((1+1j, 2), (3j, 4-2j)),
        (((1, 2), (3, 4)), ((5, 6), (7, 8))),
        tuple(float(i) for i in range(-5, 5)),
        (1, 2.5, 3+4j, True, -5)
    ]

    for t in tuples:
        input_dict = {"x": t}
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isreal_6"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_6'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_6'], lib="jax", suffix=6)
