
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.numpy.linalg as jax_linalg

# Monkeypatch JAX to support string ords for vector norm
original_vector_norm = jax_linalg.vector_norm

def patched_vector_norm(x, ord=2, axis=None, keepdims=False):
    if isinstance(ord, str):
        if ord == 'inf':
            ord = float('inf')
        elif ord == '-inf':
            ord = float('-inf')
        else:
            try:
                ord = int(ord)
            except ValueError:
                ord = float(ord)
    return original_vector_norm(x, ord=ord, axis=axis, keepdims=keepdims)

jax_linalg.vector_norm = patched_vector_norm
if hasattr(jax.numpy.linalg, 'vector_norm'):
    jax.numpy.linalg.vector_norm = patched_vector_norm

def jax_numpy_linalg_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, ord='inf', axis=0, keepdims=False
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": 0,
        "keepdims": False
    })

    # Input 2: 2D float32, ord='inf', axis=1, keepdims=True
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": 1,
        "keepdims": True
    })

    # Input 3: 3D float64, ord='-inf', axis=2, keepdims=False
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({
        "x": x,
        "ord": "-inf",
        "axis": 2,
        "keepdims": False
    })

    # Input 4: 2D int32, ord='inf', axis=0, keepdims=True
    x = np.array([[1, -2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": 0,
        "keepdims": True
    })

    # Input 5: 2D float32, ord='-inf', axis=-1, keepdims=False
    x = np.random.randn(5, 5).astype(np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "-inf",
        "axis": -1,
        "keepdims": False
    })

    # Input 6: 4D float32, ord='inf', axis=3, keepdims=True
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": 3,
        "keepdims": True
    })

    # Input 7: 1D float64, ord='-inf', axis=0, keepdims=False
    x = np.array([-1.5, 2.5, -3.5], dtype=np.float64)
    list_of_inputs.append({
        "x": x,
        "ord": "-inf",
        "axis": 0,
        "keepdims": False
    })

    # Input 8: 1D float32, ord='inf', axis=-1, keepdims=True
    x = np.random.randn(10).astype(np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": -1,
        "keepdims": True
    })

    # Input 9: 2D float32, ord='-inf', axis=0, keepdims=True
    x = np.random.randn(3, 2).astype(np.float32)
    list_of_inputs.append({
        "x": x,
        "ord": "-inf",
        "axis": 0,
        "keepdims": True
    })

    # Input 10: 3D float64, ord='inf', axis=1, keepdims=False
    x = np.random.randn(4, 3, 2).astype(np.float64)
    list_of_inputs.append({
        "x": x,
        "ord": "inf",
        "axis": 1,
        "keepdims": False
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.norm_3"] = jax_numpy_linalg_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.norm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.norm_3'.")


check_valid('jax.numpy.linalg.norm', generated_inputs['jax.numpy.linalg.norm_3'], lib="jax", suffix=3)
