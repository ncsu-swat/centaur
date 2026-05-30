
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax._src.numpy.util as jax_util
import jax._src.numpy.ufuncs as jax_ufuncs

orig_promote_args = jax_util.promote_args

def patched_promote_args(fun_name, *args):
    new_args = tuple(jnp.array(arg) if isinstance(arg, tuple) else arg for arg in args)
    return orig_promote_args(fun_name, *new_args)

jax_util.promote_args = patched_promote_args
if hasattr(jax_ufuncs, 'promote_args'):
    jax_ufuncs.promote_args = patched_promote_args

def asinh_inputs():
    list_of_inputs = []

    list_of_inputs.append({"x": (0.0, 0.5, 1.0)})
    list_of_inputs.append({"x": (-1.0, -0.5, 0.0)})
    list_of_inputs.append({"x": (1, 2, 3)})
    list_of_inputs.append({"x": ((1.0, 2.0), (3.0, 4.0))})
    list_of_inputs.append({"x": ((-1.0, -2.0), (-3.0, -4.0))})
    list_of_inputs.append({"x": (10.0, 100.0)})
    list_of_inputs.append({"x": (((0.1, 0.2),), ((0.3, 0.4),))})
    list_of_inputs.append({"x": (1.5,)})
    list_of_inputs.append({"x": (0.0,)})
    list_of_inputs.append({"x": (-5.0, 5.0)})

    return list_of_inputs

generated_inputs["jax.numpy.asinh_5"] = asinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asinh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asinh_5'.")


check_valid('jax.numpy.asinh', generated_inputs['jax.numpy.asinh_5'], lib="jax", suffix=5)
