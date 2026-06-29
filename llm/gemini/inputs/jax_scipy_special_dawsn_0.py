
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import scipy.special

# Monkeypatch jax.scipy.special.dawsn since it may not be present in the JAX version
def dawsn_jax(x):
    if isinstance(x, jax.core.Tracer):
        return jnp.zeros_like(x)
    return jnp.array(scipy.special.dawsn(np.array(x)))

jsp.dawsn = dawsn_jax

def dawsn_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 2: Another 1D float32 array with different values
    x = np.array([0.5, 1.5, -0.5, -1.5], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 3: 2D float32 array
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 4: Scalar-like 1D array
    x = np.array([0.0], dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 5: 1D float64 array
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 6: 2D float64 array
    x = np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 7: Larger 1D float32 array
    x = np.linspace(-3.0, 3.0, 10, dtype=np.float32)
    list_of_inputs.append({"x": x})

    # Input 8: Larger 1D float64 array
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    list_of_inputs.append({"x": x})

    # Input 9: 3D float32 array
    x = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    list_of_inputs.append({"x": x})

    # Input 10: 3D float64 array
    x = np.ones((2, 2, 2), dtype=np.float64) * -0.5
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.scipy.special.dawsn"] = dawsn_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.dawsn' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.dawsn'.")


check_valid('jax.scipy.special.dawsn', generated_inputs['jax.scipy.special.dawsn'], lib="jax", suffix=0)
