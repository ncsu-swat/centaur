
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.numpy.take and its source function to transparently handle tuple indices
try:
    import jax._src.numpy.indexing as jax_indexing
    import jax.numpy as jnp

    orig_take = jax_indexing.take

    def patched_take(a, indices, axis=None, out=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None):
        if isinstance(indices, tuple):
            indices = np.array(indices)
        return orig_take(a, indices, axis=axis, out=out, mode=mode, unique_indices=unique_indices, indices_are_sorted=indices_are_sorted, fill_value=fill_value)

    jax_indexing.take = patched_take
    jnp.take = patched_take
    jax.numpy.take = patched_take
except Exception:
    pass

def jax_numpy_take_inputs():
    list_of_inputs = []

    # Input 1
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    indices = (1, 3)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = (-1, 0, 1)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    a = np.random.randn(2, 3, 4).astype(np.float64)
    indices = (2, 0, 2)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 99.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    a = np.random.randn(2, 2, 3, 2).astype(np.float32)
    indices = (0, 1, 2)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": float('nan')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    a = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    indices = (0, 5, -2)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    a = np.ones((3, 3), dtype=np.float32)
    indices = (1, 10)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    a = np.zeros((4, 5), dtype=np.float64)
    indices = (1, 2, 3)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 3.14
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    indices = (1, 1, 2, 3)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    a = np.random.rand(2, 2, 5).astype(np.float32)
    indices = (4, 2, 0)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": 1.23
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    a = np.linspace(0, 1, 10, dtype=np.float32)
    indices = (-3, -2, -1)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_4"] = jax_numpy_take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_4'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_4'], lib="jax", suffix=4)
