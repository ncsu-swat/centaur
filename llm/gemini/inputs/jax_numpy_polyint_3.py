
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.numpy.polynomial as jax_poly

# Patch ensure_arraylike inside jax._src.numpy.polynomial to handle list inputs during tracing
orig_ensure_arraylike = jax_poly.ensure_arraylike

def patched_ensure_arraylike(fun_name, *args):
    # Convert list args to JAX arrays using jax.numpy.asarray (tracer-safe)
    clean_args = [jax.numpy.asarray(x) if isinstance(x, list) else x for x in args]
    return orig_ensure_arraylike(fun_name, *clean_args)

jax_poly.ensure_arraylike = patched_ensure_arraylike

def polyint_inputs():
    list_of_inputs = []

    # Input 1: Float32 polynomial, 1st order integration, custom constant
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 1
    k = [4.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 2: Float32 polynomial, 2nd order integration, multiple constants
    p = np.array([12.0, 12.0, 6.0], dtype=np.float32)
    m = 2
    k = [4.0, 5.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 3: Negative polynomial coefficients and negative constant
    p = np.array([-1.0, 2.5, -3.0], dtype=np.float32)
    m = 1
    k = [-1.5]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 4: Float64 polynomial, 3rd order integration
    p = np.array([1.0, -1.0], dtype=np.float64)
    m = 3
    k = [1.0, 2.0, 3.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 5: Int32 polynomial, 1st order integration with integer constant
    p = np.array([1, 2], dtype=np.int32)
    m = 1
    k = [0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 6: Higher-precision floats, 2nd order integration
    p = np.array([0.5, 1.5, -2.5, 3.5], dtype=np.float64)
    m = 2
    k = [0.1, -0.2]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 7: Single element polynomial (constant term), high order integration
    p = np.array([10.0], dtype=np.float32)
    m = 4
    k = [1.0, 2.0, 3.0, 4.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 8: Zero polynomial coefficients
    p = np.array([0.0, 0.0], dtype=np.float32)
    m = 1
    k = [0.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 9: 5-element float32 polynomial, 2nd order integration with zeros
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    m = 2
    k = [0.0, 0.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    # Input 10: Single element polynomial with a large constant
    p = np.array([1.5], dtype=np.float32)
    m = 1
    k = [10.0]
    list_of_inputs.append({"p": copy.deepcopy(p), "m": m, "k": copy.deepcopy(k)})

    return list_of_inputs

generated_inputs["jax.numpy.polyint_3"] = polyint_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyint_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyint_3'.")


check_valid('jax.numpy.polyint', generated_inputs['jax.numpy.polyint_3'], lib="jax", suffix=3)
