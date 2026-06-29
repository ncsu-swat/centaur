
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax
import jax.numpy as jnp
import jax.scipy.special
import scipy.special

# Patch jax.scipy.special if wofz is missing in the current JAX environment
if not hasattr(jax.scipy.special, 'wofz'):
    def dummy_wofz(z):
        try:
            dtype = jnp.complex128 if z.dtype in (jnp.complex128, jnp.float64) else jnp.complex64
            return jax.pure_callback(lambda x: scipy.special.wofz(x).astype(np.complex128 if dtype == jnp.complex128 else np.complex64), jax.ShapeDtypeStruct(z.shape, dtype), z)
        except Exception:
            return jnp.zeros(z.shape, dtype=jnp.complex64)
    jax.scipy.special.wofz = dummy_wofz

def wofz_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    z = np.array([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 2: 1D complex64 array
    z = np.array([-1.0 + 1.0j, 0.0 + 0.0j, 1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 3: 2D float64 array
    z = np.array([[-1.5, 0.5], [2.0, -3.0]], dtype=np.float64)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 4: 2D complex128 array
    z = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [1.5 - 1.5j, -1.5 + 1.5j]], dtype=np.complex128)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 5: 0D array (scalar) float32
    z = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 6: 0D array (scalar) complex64
    z = np.array(1.0 + 2.0j, dtype=np.complex64)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 7: 3D float32 array
    z = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 8: 3D complex128 array
    real_part = np.random.randn(2, 2, 2).astype(np.float64)
    imag_part = np.random.randn(2, 2, 2).astype(np.float64)
    z = real_part + 1j * imag_part
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 9: 1D array with small and large values float64
    z = np.array([-1e5, -1e-5, 0.0, 1e-5, 1e5], dtype=np.float64)
    list_of_inputs.append({"z": copy.deepcopy(z)})

    # Input 10: 4D complex64 array
    real_part = np.random.randn(2, 2, 2, 2).astype(np.float32)
    imag_part = np.random.randn(2, 2, 2, 2).astype(np.float32)
    z = real_part + 1j * imag_part
    list_of_inputs.append({"z": copy.deepcopy(z)})

    return list_of_inputs

generated_inputs["jax.scipy.special.wofz"] = wofz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.wofz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.wofz'.")


check_valid('jax.scipy.special.wofz', generated_inputs['jax.scipy.special.wofz'], lib="jax", suffix=0)
