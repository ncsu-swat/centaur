
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Monkeypatch both jax_util and jax_poly to ensure our patch is applied
# even if polynomial.py already imported ensure_arraylike.
try:
    import jax._src.numpy.util as jax_util
    import jax._src.numpy.polynomial as jax_poly
    
    orig_ensure_arraylike = jax_util.ensure_arraylike

    def patched_ensure_arraylike(fun_name, *args):
        import jax.numpy as jnp
        new_args = tuple(jnp.array(arg) if isinstance(arg, tuple) else arg for arg in args)
        return orig_ensure_arraylike(fun_name, *new_args)

    jax_util.ensure_arraylike = patched_ensure_arraylike
    jax_poly.ensure_arraylike = patched_ensure_arraylike
except Exception:
    pass

def polyint_inputs():
    list_of_inputs = []

    # Input 1: Basic first-order integration
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = (4.0,)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Second-order integration
    p = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    m = 2
    k = (4.0, 5.0)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: First-order with integer array
    p = np.array([1, 2, 3, 4], dtype=np.int32)
    m = 1
    k = (0.0,)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Third-order integration with float64
    p = np.array([5.5, -2.3], dtype=np.float64)
    m = 3
    k = (1.0, 2.0, 3.0)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: First-order negative constant
    p = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    m = 1
    k = (-1.5,)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Second-order negative constants with float64
    p = np.array([-10.0, 20.0, -30.0, 40.0], dtype=np.float64)
    m = 2
    k = (-1.0, -2.0)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Fourth-order integration
    p = np.array([1.0, -1.0], dtype=np.float32)
    m = 4
    k = (1.0, 2.0, 3.0, 4.0)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: First-order float32 with zero
    p = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    m = 1
    k = (0.0,)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Second-order decimal constants
    p = np.array([9.0, 8.0, 7.0, 6.0, 5.0], dtype=np.float32)
    m = 2
    k = (0.1, 0.2)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Third-order mixed sign constants
    p = np.array([-1.2, 3.4], dtype=np.float64)
    m = 3
    k = (-0.5, 0.5, -0.5)
    input_dict = {"p": p, "m": m, "k": k}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.polyint_4"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_4'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_4'], lib="jax", suffix=4)
