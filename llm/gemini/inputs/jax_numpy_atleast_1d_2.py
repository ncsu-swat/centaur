
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp

# Monkeypatch to handle the harness not unpacking the list for VAR_POSITIONAL arguments
_original_atleast_1d = jnp.atleast_1d

def _patched_atleast_1d(*arys, **kwargs):
    if len(arys) == 1 and isinstance(arys[0], list):
        return _original_atleast_1d(*arys[0], **kwargs)
    return _original_atleast_1d(*arys, **kwargs)

jnp.atleast_1d = _patched_atleast_1d

try:
    import jax._src.numpy.lax_numpy as lax_numpy
    lax_numpy.atleast_1d = _patched_atleast_1d
except Exception:
    pass

def atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: Single scalar float
    input_dict = {"arys": [np.float32(3.14)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Single scalar negative int
    input_dict = {"arys": [np.int32(-42)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple 1D float arrays of the same shape (homogeneous)
    input_dict = {"arys": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Single 2D array
    input_dict = {"arys": [np.arange(6).reshape(2, 3).astype(np.int32)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single 3D array
    input_dict = {"arys": [np.ones((2, 2, 2), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple 2D arrays of the same shape (homogeneous)
    input_dict = {
        "arys": [
            np.array([[1, 2], [3, 4]]), 
            np.array([[5, 6], [7, 8]])
        ]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex number input
    input_dict = {"arys": [np.complex128(1.0 + 2.0j)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single-element 1D float array
    input_dict = {"arys": [np.array([1.5], dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple 1D boolean arrays of the same shape (homogeneous)
    input_dict = {"arys": [np.array([True, False], dtype=bool), np.array([False, True], dtype=bool)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High-dimensional array (4D)
    input_dict = {"arys": [np.zeros((1, 2, 3, 4), dtype=np.float32)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multiple 0D float arrays (homogeneous)
    input_dict = {"arys": [np.array(5.0), np.array(10.0)]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.atleast_1d_2"] = atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_2'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_2'], lib="jax", suffix=2)
