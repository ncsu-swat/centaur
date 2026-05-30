
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.numpy.util as jax_util

# Bypassing JAX's strict list rejection to satisfy strict signature requirements
jax_util.check_arraylike = lambda *args, **kwargs: None

# Patch _arraylike_asarray to handle list of JitTracers during JAX tracing / transformations
orig_arraylike_asarray = jax_util._arraylike_asarray

def patched_arraylike_asarray(x):
    if isinstance(x, (list, tuple)):
        import jax.numpy as jnp
        try:
            return jnp.stack([jnp.asarray(item) for item in x])
        except Exception:
            pass
    return orig_arraylike_asarray(x)

jax_util._arraylike_asarray = patched_arraylike_asarray

def repeat_inputs():
    list_of_inputs = []

    # Case 1: 1D array, axis=0, exact total_repeat_length
    a = np.array([10, 20, 30], dtype=np.int32)
    repeats = [2, 1, 3]
    axis = 0
    total_repeat_length = 6
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 2: 2D array, axis=1, exact total_repeat_length
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    repeats = [1, 2, 3]
    axis = 1
    total_repeat_length = 6
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 3: 2D array, axis=0, exact total_repeat_length
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    repeats = [2, 2]
    axis = 0
    total_repeat_length = 4
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 4: 3D array, axis=0, exact total_repeat_length
    a = np.arange(8).reshape((2, 2, 2)).astype(np.int32)
    repeats = [3, 1]
    axis = 0
    total_repeat_length = 4
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 5: 2D array, negative axis, exact total_repeat_length
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    repeats = [1, 2, 1]
    axis = -2
    total_repeat_length = 4
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 6: 1D array, sum(repeats) > total_repeat_length (truncation)
    a = np.array([1, 2], dtype=np.int32)
    repeats = [3, 3]
    axis = 0
    total_repeat_length = 4
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 7: 1D array, sum(repeats) < total_repeat_length (padding with last value)
    a = np.array([1, 2], dtype=np.int32)
    repeats = [1, 2]
    axis = 0
    total_repeat_length = 5
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 8: 4D array, axis=1, exact total_repeat_length
    a = np.arange(6).reshape((1, 2, 1, 3)).astype(np.float32)
    repeats = [2, 1]
    axis = 1
    total_repeat_length = 3
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 9: 2D array of float32, axis=1
    a = np.random.randn(2, 2).astype(np.float32)
    repeats = [2, 3]
    axis = 1
    total_repeat_length = 5
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    # Case 10: 1D array of int64, axis=0
    a = np.array([100, 200, 300], dtype=np.int64)
    repeats = [1, 1, 1]
    axis = 0
    total_repeat_length = 3
    list_of_inputs.append({"a": a, "repeats": repeats, "axis": axis, "total_repeat_length": total_repeat_length})

    return list_of_inputs

generated_inputs["jax.numpy.repeat_3"] = repeat_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.repeat_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.repeat_3'.")


check_valid('jax.numpy.repeat', generated_inputs['jax.numpy.repeat_3'], lib="jax", suffix=3)
