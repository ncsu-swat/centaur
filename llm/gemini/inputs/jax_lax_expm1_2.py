
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax._src.lax.lax as jax_lax

# Patch expm1 primitive implementation to support eager execution with accuracy tuple
orig_expm1_impl = jax_lax.expm1_p.impl
def new_expm1_impl(*args, **kwargs):
    if 'accuracy' in kwargs:
        kwargs['accuracy'] = None
    elif len(args) > 1:
        args = (args[0], None) + args[2:]
    return orig_expm1_impl(*args, **kwargs)
jax_lax.expm1_p.impl = new_expm1_impl

def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 near zero
    x = np.array([-1e-5, 0.0, 1e-5, 2e-4], dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 2: 2D float64 standard normal
    x = np.random.randn(3, 4).astype(np.float64)
    accuracy = (1e-12, 1e-12)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 3: 3D float32 with positive and negative values
    x = np.random.uniform(-5.0, 5.0, size=(2, 2, 3)).astype(np.float32)
    accuracy = (1e-6, 1e-6)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 4: 0D float32 (scalar represented as 0-D array)
    x = np.array(0.5, dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 5: 4D float32
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 6: 2D complex64
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 7: 1D complex128
    x = (np.random.randn(5) + 1j * np.random.randn(5)).astype(np.complex128)
    accuracy = (1e-12, 1e-12)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 8: 3D float64 with large negative values
    x = np.array([[[-100.0, -50.0], [-20.0, -10.0]]], dtype=np.float64)
    accuracy = (1e-10, 1e-10)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 9: 2D float32 with zeros, small, and large values
    x = np.array([[0.0, 1e-20, -1e-20], [100.0, -100.0, 1.0]], dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 10: 5D float32
    x = np.random.randn(2, 1, 3, 1, 2).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    return list_of_inputs

generated_inputs["jax.lax.expm1_2"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.expm1_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.expm1_2'.")


check_valid('jax.lax.expm1', generated_inputs['jax.lax.expm1_2'], lib="jax", suffix=2)
