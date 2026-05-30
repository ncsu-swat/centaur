
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax._src.numpy.util as jax_util
import jax._src.numpy.ufuncs as jax_ufuncs

# Monkey-patch JAX to convert tuple arguments to JAX arrays before processing
_original_promote_args_inexact = jax_util.promote_args_inexact

def _patched_promote_args_inexact(fun_name, *args):
    new_args = [jnp.asarray(arg) if isinstance(arg, tuple) else arg for arg in args]
    return _original_promote_args_inexact(fun_name, *new_args)

jax_util.promote_args_inexact = _patched_promote_args_inexact
if hasattr(jax_ufuncs, "promote_args_inexact"):
    jax_ufuncs.promote_args_inexact = _patched_promote_args_inexact

def arccosh_inputs():
    list_of_inputs = []

    # Input 1: Simple float tuple
    list_of_inputs.append({"x": (1.0, 2.0, 5.0, 10.0)})

    # Input 2: Single-element float tuple
    list_of_inputs.append({"x": (1.5,)})

    # Input 3: Tuple of integers
    list_of_inputs.append({"x": (2, 5, 10, 50, 100)})

    # Input 4: Tuple of complex numbers
    list_of_inputs.append({"x": (1.0 + 2.0j, 2.0 + 3.0j, 5.0 + 0j)})

    # Input 5: Nested 2D tuple
    list_of_inputs.append({"x": ((2.0, 3.0), (4.0, 5.0))})

    # Input 6: Values close to 1
    list_of_inputs.append({"x": (1.001, 1.01, 1.1, 1.5)})

    # Input 7: High dimensional nested tuple (3D)
    list_of_inputs.append({"x": (((2.0, 3.0), (4.0, 5.0)), ((6.0, 7.0), (8.0, 9.0)))})

    # Input 8: Tuple of numpy float32 scalars
    list_of_inputs.append({"x": (np.float32(2.0), np.float32(3.0))})

    # Input 9: Large scale values
    list_of_inputs.append({"x": (1e5, 1e10, 1e15)})

    # Input 10: Mixed integer-like float values
    list_of_inputs.append({"x": (2.0, 3.0, 4.0)})

    return list_of_inputs

generated_inputs["jax.numpy.arccosh_5"] = arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_5'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_5'], lib="jax", suffix=5)
