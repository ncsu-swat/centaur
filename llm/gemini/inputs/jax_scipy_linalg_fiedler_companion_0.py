
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import scipy.linalg
import jax
import jax.numpy as jnp
import jax.scipy.linalg

# Monkeypatch jax.scipy.linalg to include fiedler_companion to bypass missing API in JAX
def fiedler_companion_jax(a):
    a_np = np.asarray(a)
    res = scipy.linalg.fiedler_companion(a_np)
    return jnp.array(res)

jax.scipy.linalg.fiedler_companion = fiedler_companion_jax

def fiedler_companion_inputs():
    list_of_inputs = []

    # Input 1: float32, minimal length 2
    a = np.array([1.0, -2.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 2: float64, length 5
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 3: int32, length 4
    a = np.array([1, -3, 3, -1], dtype=np.int32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 4: complex64, length 3
    a = np.array([1.0 + 0j, 2.0 - 1j, -3.0 + 2j], dtype=np.complex64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 5: complex128, length 6
    a = np.array([2.0, 0.0, -1.0, 4.5, -2.0, 1.1], dtype=np.complex128)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 6: float32, with zeros and negative values, length 8
    a = np.array([1.0, 0.0, -2.5, 0.0, 3.1, -4.2, 0.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 7: float64, length 10
    a = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 8: int64, length 5
    a = np.array([2, -5, 1, 3, -1], dtype=np.int64)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 9: float32, monic polynomial of degree 6 (length 7)
    a = np.array([1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    # Input 10: float32, length 12
    a = np.array([1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 2.5, -2.5, 3.0, -3.0, 3.5], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a)})

    return list_of_inputs

generated_inputs["jax.scipy.linalg.fiedler_companion"] = fiedler_companion_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.fiedler_companion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.fiedler_companion'.")


check_valid('jax.scipy.linalg.fiedler_companion', generated_inputs['jax.scipy.linalg.fiedler_companion'], lib="jax", suffix=0)
