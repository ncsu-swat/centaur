
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.indexing as jax_indexing

_original_put = jnp.put

def patched_put(a, ind, v, mode=None, inplace=True):
    if isinstance(v, tuple):
        v = np.array(v)
    return _original_put(a, ind, v, mode=mode, inplace=inplace)

jnp.put = patched_put
jax_indexing.put = patched_put

def jax_numpy_put_inputs():
    list_of_inputs = []

    # Input 1
    a = np.zeros(5, dtype=np.float32)
    ind = np.array([0, 2, 4], dtype=np.int32)
    v = (10.0, 20.0, 30.0)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    # Input 2
    a = np.ones((3, 3), dtype=np.int32)
    ind = np.array([1, 4, 7], dtype=np.int32)
    v = (5, 6, 7)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    })

    # Input 3
    a = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
    ind = np.array([0, 5, 11], dtype=np.int64)
    v = (-1.0, -2.0, -3.0)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    # Input 4
    a = np.zeros(6, dtype=np.int32)
    ind = np.array([1, 3, 10], dtype=np.int32)
    v = (100, 200, 300)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    # Input 5
    a = np.zeros(6, dtype=np.int32)
    ind = np.array([1, 3, 10], dtype=np.int32)
    v = (100, 200, 300)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    })

    # Input 6
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ind = np.array([-1, -3], dtype=np.int32)
    v = (99.0, 88.0)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    # Input 7
    a = np.array([True, False, True, False], dtype=bool)
    ind = np.array([1, 3], dtype=np.int32)
    v = (True, True)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    })

    # Input 8
    a = np.random.randn(10).astype(np.float32)
    ind = np.array([2, 5, 8], dtype=np.int32)
    v = (4.2,)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    # Input 9
    a = np.arange(10).astype(np.int32)
    ind = np.array([[1, 2], [3, 4]], dtype=np.int32)
    v = (10, 20, 30, 40)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "wrap",
        "inplace": False
    })

    # Input 10
    a = np.ones((2, 2, 2, 2), dtype=np.float32)
    ind = np.array([0, 15], dtype=np.int32)
    v = (0.0, 0.0)
    list_of_inputs.append({
        "a": a,
        "ind": ind,
        "v": v,
        "mode": "clip",
        "inplace": False
    })

    return list_of_inputs

generated_inputs["jax.numpy.put_5"] = jax_numpy_put_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_5'.")


check_valid('jax.numpy.put', generated_inputs['jax.numpy.put_5'], lib="jax", suffix=5)
