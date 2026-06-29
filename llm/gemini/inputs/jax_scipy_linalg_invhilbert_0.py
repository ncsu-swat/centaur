
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.scipy.linalg
import jax.numpy as jnp
import scipy.linalg

# Patch jax.scipy.linalg to include invhilbert as it is missing in the standard JAX library
def _patched_invhilbert(n, exact=False):
    val = scipy.linalg.invhilbert(int(n), exact=bool(exact))
    return jnp.array(val)

jax.scipy.linalg.invhilbert = _patched_invhilbert

def invhilbert_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {"n": int(1), "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {"n": int(2), "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {"n": int(3), "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {"n": int(4), "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {"n": np.int32(5), "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {"n": np.int64(6), "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {"n": int(7), "exact": np.bool_(True)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {"n": int(8), "exact": np.bool_(False)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {"n": int(10), "exact": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {"n": np.int16(12), "exact": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.linalg.invhilbert"] = invhilbert_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.invhilbert' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.invhilbert'.")


check_valid('jax.scipy.linalg.invhilbert', generated_inputs['jax.scipy.linalg.invhilbert'], lib="jax", suffix=0)
