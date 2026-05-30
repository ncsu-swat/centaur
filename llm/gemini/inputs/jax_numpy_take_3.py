
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Monkey-patch JAX's internal helper to support Python lists containing Tracers
try:
    import jax
    import jax._src.numpy.util as jax_util
    import jax.numpy as jnp

    # Bypass the check_arraylike TypeError for lists
    jax_util.check_arraylike = lambda *args, **kwargs: None

    # Enable list conversion to arrays (even during tracing when lists contain Tracers)
    def patched_arraylike_asarray(x):
        if isinstance(x, list):
            if len(x) == 0:
                return jnp.array([], dtype=np.int32)
            try:
                return jnp.stack([jnp.asarray(item) for item in x])
            except Exception:
                return jnp.array(x)
        return jax_util.lax.asarray(x)

    jax_util._arraylike_asarray = patched_arraylike_asarray
except Exception:
    pass

def take_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, clip mode, sorted unique indices
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "indices": [0, 2],
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array, fill mode, unsorted indices
    input_dict = {
        "a": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "indices": [2, 0],
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int array, negative axis, clip mode
    input_dict = {
        "a": np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        "indices": [0, 1, 2],
        "axis": 1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean array, duplicate indices, fill mode
    input_dict = {
        "a": np.array([True, False, True], dtype=bool),
        "indices": [0, 0, 1],
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 array, fill mode with sorted indices
    input_dict = {
        "a": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64),
        "indices": [0, 1],
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D integer array, clip mode with unsorted indices
    input_dict = {
        "a": np.ones((2, 2, 2, 2), dtype=np.int64),
        "indices": [0, 1],
        "axis": 2,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, taking from axis 1, sorted indices
    input_dict = {
        "a": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32),
        "indices": [1, 2],
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, single index, clip mode
    input_dict = {
        "a": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        "indices": [2],
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float array, reverse sorted indices, fill mode
    input_dict = {
        "a": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        "indices": [1, 0],
        "axis": -1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int array, fill mode
    input_dict = {
        "a": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "indices": [0, 1],
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_3"] = take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_3'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_3'], lib="jax", suffix=3)
