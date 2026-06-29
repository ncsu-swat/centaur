
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

# Monkeypatch jax.lax.cos_p to allow passing tuple accuracy on CPU
if hasattr(jax.lax, 'cos_p'):
    orig_bind = jax.lax.cos_p.bind
    def new_bind(*args, **kwargs):
        if 'accuracy' in kwargs:
            kwargs['accuracy'] = None
        return orig_bind(*args, **kwargs)
    jax.lax.cos_p.bind = new_bind

def jax_lax_cos_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([0.0, np.pi/4, np.pi/2, np.pi], dtype=np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 2: 2D float32 array with negative and positive values
    x = np.random.uniform(-10.0, 10.0, size=(3, 5)).astype(np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 3: 3D float64 array
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, size=(2, 3, 4)).astype(np.float64)
    accuracy = (1e-8, 1e-8)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 4: 1D complex64 array
    x = np.array([1.0 + 1j, -2.0 + 0.5j, 3.0j], dtype=np.complex64)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 5: Scalar (0D array) float32
    x = np.array(1.5, dtype=np.float32)
    accuracy = (1e-3, 1e-3)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 6: 2D complex128 array
    real = np.random.uniform(-5.0, 5.0, size=(2, 2))
    imag = np.random.uniform(-5.0, 5.0, size=(2, 2))
    x = (real + 1j * imag).astype(np.complex128)
    accuracy = (1e-12, 1e-12)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 7: Large 4D float32 array
    x = np.random.normal(0.0, 1.0, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = (1e-5, 1e-5)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 8: 1D array with special float values (zeros, small numbers)
    x = np.array([0.0, -0.0, 1e-15, -1e-15], dtype=np.float32)
    accuracy = (1e-6, 1e-6)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 9: 5D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    accuracy = (1e-4, 1e-4)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    # Input 10: 2D float64 array with extremely large values
    x = np.array([[1e6, -1e6], [1e5, -1e5]], dtype=np.float64)
    accuracy = (1e-7, 1e-7)
    list_of_inputs.append({"x": x, "accuracy": accuracy})

    return list_of_inputs

generated_inputs["jax.lax.cos_2"] = jax_lax_cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cos_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cos_2'.")


check_valid('jax.lax.cos', generated_inputs['jax.lax.cos_2'], lib="jax", suffix=2)
